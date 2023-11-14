from datasets import load_dataset
from src.utils import (
    prepare_text,
    row_to_string,
    multiple_row_to_string,
)
from src.utils import legacy_get_dataset_info

from transformers import AutoTokenizer
import numpy as np
import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
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
from src.utils import (
    prepare_text,
    row_to_string,
    multiple_row_to_string,
)
from src.utils import legacy_get_dataset_info

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="wine_30",
    help="Name of config from the the multi_config.yaml file",
)
config_type = parser.parse_args().config

"""
name = "kick"
di = get_dataset_info(name, model_type="all_as_text")
dataset = load_dataset(di.ds_name)  # , download_mode="force_redownload")
dataset = prepare_text(dataset, "all_as_text", name)

imdb = dataset

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")


# old_imdb = load_dataset("imdb")
# def preprocess_function(examples):
#     return tokenizer(examples["text"], truncation=True)
# old_tokenized_imdb = old_imdb.map(preprocess_function, batched=True)


# tokenized_imdb = imdb.map(preprocess_function, batched=True)


def encode(examples):
    return {
        # "label": np.array([examples[di.label_col]]),
        "label": examples[di.label_col],
        **tokenizer(examples["text"], truncation=True, padding="max_length"),
    }


new_tokenized_imdb = imdb.map(encode, load_from_cache_file=False)


# accuracy = evaluate.load("accuracy")


# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
#     return accuracy.compute(predictions=predictions, references=labels)


model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    num_labels=2,
    # id2label=id2label,
    # label2id=label2id
)

training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    # push_to_hub=True,
    # torch_compile=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=new_tokenized_imdb["train"],
    eval_dataset=new_tokenized_imdb["test"],
    tokenizer=tokenizer,
    # data_collator=data_collator,
    # compute_metrics=compute_metrics,
)
"""
###
# Import yaml file
with open("configs/train_default.yaml") as f:
    args = yaml.safe_load(f)

# Update default args with chosen config
if config_type != "default":
    with open("configs/train_configs.yaml") as f:
        yaml_configs = yaml.safe_load_all(f)
        yaml_args = next(conf for conf in yaml_configs if conf["config"] == config_type)
    args.update(yaml_args)
    print(f"Updating with:\n{yaml_args}\n")
print(f"\n{args}\n")

# Dataset
di = legacy_get_dataset_info(args["dataset"], model_type=args["version"])
dataset = load_dataset(di.ds_name)  # , download_mode="force_redownload")
dataset = prepare_text(dataset, args["version"], args["dataset"])
if di.prob_type == "regression":
    mean_price = np.mean(dataset["train"]["label"])
    std_price = np.std(dataset["train"]["label"])

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    args["model_base"], num_labels=di.num_labels, problem_type=di.prob_type
)
tokenizer = AutoTokenizer.from_pretrained(args["model_base"])


# Tokenize the dataset
def encode(examples):
    return {
        "label": np.array([examples[di.label_col]]),
        # "label": examples[di.label_col],
        **tokenizer(examples["text"], truncation=True, padding="max_length"),
    }


dataset = dataset.map(encode, load_from_cache_file=True)

# Fast dev run if want to run quickly and not save to wandb
if args["fast_dev_run"]:
    args["num_epochs"] = 1
    args["tags"].append("fast-dev-run")
    dataset["train"] = dataset["train"].select(range(500))
    dataset["test"] = dataset["test"].select(range(10))
    output_dir = os.path.join(args["output_root"], "testing")
    print(
        "\n######################    Running in fast dev mode    #######################\n"
    )

# If not, initialize wandb
else:
    wandb.init(
        project=di.wandb_proj_name,
        tags=args["tags"],
        save_code=True,
        config={"my_args/" + k: v for k, v in args.items()},
    )
    os.environ["WANDB_LOG_MODEL"] = "True"
    output_dir = os.path.join(args["output_root"], args["dataset"], wandb.run.name)
    print(f"Results will be saved @: {output_dir}")

# Make output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save args file
with open(os.path.join(output_dir, "args.yaml"), "w") as f:
    yaml.dump(args, f)

# Initialise training arguments and trainer
# training_args = TrainingArguments(
#     output_dir=output_dir,
#     num_train_epochs=args["num_epochs"],
#     per_device_train_batch_size=args["batch_size"],
#     per_device_eval_batch_size=args["batch_size"],
#     logging_steps=args["logging_steps"],
#     learning_rate=args["lr"],
#     weight_decay=args["weight_decay"],
#     gradient_accumulation_steps=args["grad_accumulation_steps"],
#     warmup_ratio=args["warmup_ratio"],
#     lr_scheduler_type=args["lr_scheduler"],
#     dataloader_num_workers=args["num_workers"],
#     do_train=args["do_train"],
#     do_predict=args["do_predict"],
#     resume_from_checkpoint=args["resume_from_checkpoint"],
#     report_to="wandb" if not args["fast_dev_run"] else "none",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     save_total_limit=args["save_total_limit"],
#     load_best_model_at_end=True,
#     torch_compile=args["pytorch2.0"],  # Needs to be true if PyTorch 2.0
# )

# if args["lion_optim"]:
#     opt = Lion(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
#     sched = None

# trainer = Trainer(
#     model=model,
#     optimizers=(opt, sched) if args["lion_optim"] else (None, None),
#     args=training_args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["validation"],
#     callbacks=[EarlyStoppingCallback(args["early_stopping_patience"])]
#     if args["early_stopping_patience"] > 0
#     else [],
# )
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=args["num_epochs"],
    per_device_train_batch_size=32,  # args["batch_size"],
    per_device_eval_batch_size=2,  # args["batch_size"],
    logging_steps=args["logging_steps"],
    learning_rate=args["lr"],
    weight_decay=args["weight_decay"],
    gradient_accumulation_steps=args["grad_accumulation_steps"],
    warmup_ratio=args["warmup_ratio"],
    lr_scheduler_type=args["lr_scheduler"],
    dataloader_num_workers=args["num_workers"],
    do_train=args["do_train"],
    do_predict=args["do_predict"],
    resume_from_checkpoint=args["resume_from_checkpoint"],
    report_to="wandb" if not args["fast_dev_run"] else "none",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=args["save_total_limit"],
    load_best_model_at_end=True,
    # torch_compile=args["pytorch2.0"],  # Needs to be true if PyTorch 2.0
)

trainer = Trainer(
    model=model,
    # optimizers=(opt, sched) if args["lion_optim"] else (None, None),
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    callbacks=[EarlyStoppingCallback(args["early_stopping_patience"])]
    if args["early_stopping_patience"] > 0
    else [],
)
trainer.train()
