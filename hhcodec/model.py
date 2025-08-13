import torch
import torch.nn.functional as F
import lightning as L
import math
from contextlib import contextmanager

from hhcodec.neuralmodel.seanet import SEANetEncoder as Encoder
from hhcodec.neuralmodel.seanet import SEANetDecoder as Decoder
from hhcodec.neuralmodel.vocosbackbone import VocosBackbone as Backbone
from hhcodec.lr_scheduler import Scheduler_LinearWarmup, Scheduler_LinearWarmup_CosineDecay
from hhcodec.util import requires_grad,LitEma,instantiate_from_config
from hhcodec.vq import ResidualSimVQ
import bigvgan
from collections import OrderedDict
from einops import rearrange
import torch.nn as nn


class melHead(nn.Module):
    def __init__(self):
        super(melHead, self).__init__() 
        self.transconv = nn.ConvTranspose1d(
            in_channels=768, 
            out_channels=100,  
            kernel_size=4,                
            stride=4,  
            padding=0,
            output_padding=0,
        )

    def forward(self, x):
        x = x.transpose(1, 2)  
        x = self.transconv(x) 
        return x

class VQModel(L.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 ### Quantize Related
                 quantconfig,
                 sample_rate,
                 target_bandwidths,
                 audio_normalize,
                 segment,
                 ckpt_path=None,
                 ignore_keys=[],
                 colorize_nlabels=None,
                 monitor=None,
                 learning_rate=None,
                 ### scheduler config
                 warmup_epochs=1.0, #warmup epochs
                 scheduler_type = "linear-warmup_cosine-decay",
                 min_learning_rate = 0,
                 use_ema = False,
                 stage = None,
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.backbone = Backbone(
            input_channels=512, 
            dim=768,
            intermediate_dim=2304,
            num_layers=12,
            adanorm_num_embeddings=None # only one quantizer layer so no adanorm required
        )
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels=512,
            out_channels=768,
            kernel_size=11,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.melhead = melHead()
        self.loss = instantiate_from_config(lossconfig)
        
        self.audio_normalize = audio_normalize
        self.bigvqgan = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x', use_cuda_kernel=False)

        self.quantize = ResidualSimVQ(
            dim = 512,
            codebook_size = 8192,
            rotation_trick = True,  
            codebook_transform = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512)
            ),
            channel_first= True,
            num_quantizers=2,
        )

        self.use_ema = use_ema
        self.stage = stage
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, stage=stage)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        if self.use_ema and stage is None: #no need to construct ema when training transformer
            self.model_ema = LitEma(self)
        self.learning_rate = learning_rate
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.min_learning_rate = min_learning_rate
        self.automatic_optimization = False
        self.strict_loading = False

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def get_ema_model(self):
        if self.use_ema:
            self.model_ema.copy_to(self)  # 直接应用 EMA 参数
        return self  # 返回修改后的模型

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        '''
        save the state_dict and filter out the 
        '''
        return {k: v for k, v in super().state_dict(*args, destination, prefix, keep_vars).items() if ("inception_model" not in k and "lpips_vgg" not in k and "lpips_alex" not in k)}
        
    def init_from_ckpt(self, path, ignore_keys=list(), stage=None):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        ema_mapping = {}
        new_params = OrderedDict()
        if stage == "transformer": ### directly use ema encoder and decoder parameter
            if self.use_ema:
                for k, v in sd.items(): 
                    if "encoder" in k:
                        if "model_ema" in k:
                            k = k.replace("model_ema.", "") #load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v   
                        s_name = k.replace('.', '')
                        ema_mapping.update({s_name: k})
                        continue
                    if "decoder" in k:
                        if "model_ema" in k:
                            k = k.replace("model_ema.", "") #load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v 
                        s_name = k.replace(".", "")
                        ema_mapping.update({s_name: k})
                        continue 
            else: #also only load the Generator
                for k, v in sd.items():
                    if "encoder" in k:
                        new_params[k] = v
                    elif "decoder" in k:
                        new_params[k] = v
            missing_keys, unexpected_keys = self.load_state_dict(new_params, strict=False)
        else: ## simple resume
            missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        if self.audio_normalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None
        
        h = self.encoder(x)
        # (quant, emb_loss, info), loss_breakdown = self.quantize(h)
        quant, info, loss_breakdown,first_quant,second_quant,first_index = self.quantize(h)
        return (quant, scale), torch.tensor(0.0), info, loss_breakdown,first_quant,second_quant,first_index

    def decode(self, quant):
        dec = self.backbone(quant)
        # dec = self.VideoConv(dec)
        mel = self.melhead(dec)
        dec = self.bigvqgan(mel)
        return mel,dec

    def forward(self, input):
        quant, diff, indices, loss_break,first_quant,second_quant,first_index = self.encode(input)
        loss_break=sum(loss_break)
        mel,dec = self.decode(first_quant)

        for ind in indices.unique():
            self.codebook_count[ind] = 1
        # feature = rearrange(quant[0], 'b d t -> b t d')
        # feature = self.transform(feature)
        # feature = rearrange(feature, 'b t d -> b d t')
        feature = self.conv_transpose(first_quant)
        feature = rearrange(feature, 'b d t -> b t d')
        return mel,dec, diff, loss_break,feature
    
    def d_axis_distill_loss(self,feature, target_feature):
        n = min(feature.size(1), target_feature.size(1))
        distill_loss = - torch.log(torch.sigmoid(torch.nn.functional.cosine_similarity(feature[:, :n], target_feature[:, :n], axis=1))).mean()
        return distill_loss

    def t_axis_distill_loss(self,feature, target_feature, lambda_sim=1):
        n = min(feature.size(1), target_feature.size(1))
        l1_loss = F.l1_loss(feature[:, :n], target_feature[:, :n], reduction='mean')
        sim_loss = - torch.log(torch.sigmoid(torch.nn.functional.cosine_similarity(feature[:, :n], target_feature[:, :n], axis=-1))).mean()
        distill_loss = l1_loss + lambda_sim * sim_loss
        return distill_loss 
        # def get_input(self, batch):
        #     x = batch["waveform"].to(memory_format=torch.contiguous_format)
        #     return x.float()

    # fix mulitple optimizer bug
    # refer to https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
    def training_step(self, batch, batch_idx):
        x,semantic_feature,mel=batch[0],batch[1],batch[2]
        x = x.unsqueeze(1)
        mel_rec,xrec, eloss, loss_break,feature = self(x)

        opt_gen, opt_disc = self.optimizers()
        # scheduler_gen, scheduler_disc = self.lr_schedulers()
        if self.scheduler_type != "None":
            scheduler_gen,scheduler_disc = self.lr_schedulers()
        ####################
        # fix global step bug
        # refer to https://github.com/Lightning-AI/pytorch-lightning/issues/17958
        opt_disc._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        opt_disc._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        # opt_gen._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        # opt_gen._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        ####################
        # optimize generator
        loss_distill = self.d_axis_distill_loss(feature, semantic_feature)
        aeloss, log_dict_ae = self.loss(loss_distill,eloss, loss_break,mel,mel_rec ,x, xrec, 0, self.global_step,split="train")
        opt_gen.zero_grad()
        self.manual_backward(aeloss)
        opt_gen.step()
        # scheduler_gen.step()
        if self.scheduler_type != "None":
            scheduler_gen.step()
        log_dict_ae["train/codebook_util"] = torch.tensor(sum(self.codebook_count) / len(self.codebook_count))
        
        # optimize discriminator
        discloss, log_dict_disc = self.loss(loss_distill,eloss, loss_break, mel,mel_rec,x, xrec, 1, self.global_step,
                                            split="train")
        opt_disc.zero_grad()
        self.manual_backward(discloss)
        opt_disc.step()
        # scheduler_disc.step()
        if self.scheduler_type != "None":
            scheduler_disc.step()
        if torch.distributed.get_rank() == 0:
            print(log_dict_ae, log_dict_disc)
        
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema and self.current_epoch >= 10:
            self.model_ema(self)
            
    def on_train_epoch_start(self):
        self.codebook_count = [0] * 8192
        if self.current_epoch < 10:
            for param in self.bigvqgan.parameters():
                param.requires_grad = False
        else:
            for param in self.bigvqgan.parameters():
                param.requires_grad = True
        
    def on_validation_epoch_start(self):
        self.codebook_count = [0] * 8192

    def validation_step(self, batch, batch_idx): 
        if self.use_ema:
            with self.ema_scope():
                log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        else:
            log_dict = self._validation_step(batch, batch_idx)

    def _validation_step(self, batch, batch_idx, suffix=""):
        x,semantic_feature=batch[0],batch[1]
        x = x.unsqueeze(1)
        quant, eloss, indices, loss_break = self.encode(x)
        x_rec = self.decode(quant)
        feature = rearrange(quant[0], 'b d t -> b t d')
        feature = self.transform(feature)
        loss_distill = self.d_axis_distill_loss(feature, semantic_feature)
        aeloss, log_dict_ae = self.loss(loss_distill,eloss, loss_break, x, x_rec, 0, self.global_step,
                                        split="val"+ suffix)
        discloss, log_dict_disc = self.loss(loss_distill,eloss, loss_break, x, x_rec, 1, self.global_step,
                                            split="val" + suffix)
        for ind in indices.unique():
            self.codebook_count[ind] = 1
        log_dict_ae[f"val{suffix}/codebook_util"] = torch.tensor(sum(self.codebook_count) / len(self.codebook_count))
    
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_gen = torch.optim.Adam(
                    list(self.encoder.parameters()) +
                    list(self.decoder.parameters()) +
                    list(self.conv_transpose.parameters()) +
                    list(self.quantize.parameters()) +
                    list(self.backbone.parameters()) +
                    list(self.melhead.parameters()) +
                    list(self.bigvqgan.parameters()), 
                    lr=lr,
                    betas=(0.5, 0.9)
                )
        opt_disc = torch.optim.Adam(list(self.loss.multiperioddisc.parameters())+
                                     list(self.loss.multiresddisc.parameters())+
                                     list(self.loss.dac.parameters()),
                                    lr=lr, betas=(0.5, 0.9))
        if self.trainer.is_global_zero:
            print("step_per_epoch: {}".format(len(self.trainer.datamodule.train_dataloader()) // self.trainer.world_size))
        step_per_epoch  = len(self.trainer.datamodule.train_dataloader()) // self.trainer.world_size
        warmup_steps = step_per_epoch * self.warmup_epochs
        training_steps = step_per_epoch * self.trainer.max_epochs

        if self.scheduler_type == "None":
            return ({"optimizer": opt_gen}, {"optimizer": opt_disc})
    
        if self.scheduler_type == "linear-warmup":
            scheduler_ae = torch.optim.lr_scheduler.LambdaLR(opt_gen, Scheduler_LinearWarmup(warmup_steps))
            scheduler_disc = torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup(warmup_steps))

        elif self.scheduler_type == "linear-warmup_cosine-decay":
            multipler_min = self.min_learning_rate / self.learning_rate
            scheduler_ae = torch.optim.lr_scheduler.LambdaLR(opt_gen, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=training_steps, multipler_min=multipler_min))
            scheduler_disc = torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=training_steps, multipler_min=multipler_min))
        else:
            raise NotImplementedError()
        return {"optimizer": opt_gen, "lr_scheduler": scheduler_ae}, {"optimizer": opt_disc, "lr_scheduler": scheduler_disc}
