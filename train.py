import argparse, os, sys, datetime, glob, importlib
from torch.utils.data import random_split, DataLoader, Dataset

import lightning as L
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from lightning import seed_everything
from swanlab.integration.pytorch_lightning import SwanLabLogger
from torch.utils.data.dataloader import default_collate as custom_collate
import torch
torch.backends.cudnn.deterministic = True #True
torch.backends.cudnn.benchmark = False #False


def main():
    cli = LightningCLI(
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
