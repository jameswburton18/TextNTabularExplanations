from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_from_disk, load_dataset
import wandb
import os
import yaml
import argparse
from transformers.trainer_callback import EarlyStoppingCallback
import evaluate
import numpy as np
from lion_pytorch import Lion
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import AdamW
from utils import (
    prepare_text,
    row_to_string,
    multiple_row_to_string,
)
from src.dataset_info import get_dataset_info

names = ["jigsaw_10", "fake_10", "kick_10", "wine_10"]
for model, name in zip(
    [
        "models/jigsaw/stoic-blaze-10/checkpoint-2657",
        "models/fake/avid-dream-14/checkpoint-676",
        "models/kick/comfy-plant-14/checkpoint-2298",
        "models/wine/flowing-cherry-13/checkpoint-6705",
    ],
    names,
):
    m = AutoModelForSequenceClassification.from_pretrained(model)
    m.push_to_hub(name, private=True)
