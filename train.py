import argparse
import datetime
import glob
import importlib
import os
import sys

import lightning as L
import torch
from lightning import seed_everything
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from swanlab.integration.pytorch_lightning import SwanLabLogger
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.dataloader import default_collate as custom_collate

torch.backends.cudnn.deterministic = True #True
torch.backends.cudnn.benchmark = False #False


def main():
    cli = LightningCLI(
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
