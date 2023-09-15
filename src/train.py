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
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import AdamW
from utils import (
    prepare_text,
    ConfigLoader,
)
from src.dataset_info import get_dataset_info

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="testing",
    help="Name of config from the the multi_config.yaml file",
)
config_type = parser.parse_args().config


def main():
    # Training args
    args = ConfigLoader(
        config_type, "configs/train_configs.yaml", "configs/train_default.yaml"
    ).__dict__

    # Dataset info
    di = ConfigLoader(args["dataset"], "configs/dataset_configs.yaml")
    # Datasets which use the datasets which are all as strings
    all_text_versions = [
        "all_as_text",
        "all_as_text_base_reorder",
        "all_as_text_tnt_reorder",
    ]
    ds_name = (
        di.all_text_dataset
        if args["version"] in all_text_versions
        else di.ordinal_dataset
    )
    dataset = load_dataset(ds_name)  # , download_mode="force_redownload")
    dataset = prepare_text(
        dataset=dataset,
        di=di,
        version=args["version"],
        model_name=args["model_base"],
    )

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        args["model_base"],
        num_labels=di.num_labels,
    )
    tokenizer = AutoTokenizer.from_pretrained(args["model_base"])

    # Tokenize the dataset
    def encode(examples):
        return {
            "label": np.array([examples[di.label_col]]),
            **tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=args["max_length"],
            ),
        }

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
    # , load_from_cache_file=True)

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

    dataset = dataset.map(encode)

    # Make output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save args file
    with open(os.path.join(output_dir, "args.yaml"), "w") as f:
        yaml.dump(args, f)

    # Initialise training arguments and trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args["num_epochs"],
        per_device_train_batch_size=args["batch_size"],
        per_device_eval_batch_size=args["batch_size"],
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
        seed=args["seed"],
        torch_compile=args["pytorch2.0"],  # Needs to be true if PyTorch 2.0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        callbacks=[EarlyStoppingCallback(args["early_stopping_patience"])]
        if args["early_stopping_patience"] > 0
        else [],
    )

    # Train model
    if args["do_train"]:
        print("Training...")
        trainer.train()
        if not args["fast_dev_run"]:
            model.push_to_hub(config_type, private=True)
        print("Training complete")

    # Predict on the test set
    if args["do_predict"]:
        print("***** Running Prediction *****")
        # Test the model
        results = trainer.evaluate(dataset["test"], metric_key_prefix="test")
        preds = trainer.predict(dataset["test"]).predictions
        labels = [l[0] for l in dataset["test"]["label"]]
        if di.num_labels == 2:
            results["test/accuracy"] = np.mean(np.argmax(preds, axis=1) == labels)
            results["test/precision"] = precision_score(
                labels,
                np.argmax(preds, axis=1),
                labels=np.arange(di.num_labels),
                zero_division=0,
            )
            results["test/recall"] = recall_score(
                labels,
                np.argmax(preds, axis=1),
                labels=np.arange(di.num_labels),
                zero_division=0,
            )
            results["test/roc_auc"] = roc_auc_score(labels, preds[:, 1])
        elif di.num_labels > 2:
            results["test/accuracy"] = np.mean(np.argmax(preds, axis=1) == labels)
            results["test/precision"] = precision_score(
                labels,
                np.argmax(preds, axis=1),
                average="macro",
                labels=np.arange(di.num_labels),
                zero_division=0,
            )
            results["test/recall"] = recall_score(
                labels,
                np.argmax(preds, axis=1),
                average="macro",
                labels=np.arange(di.num_labels),
                zero_division=0,
            )

        # Save the predictions
        with open(os.path.join(output_dir, "test_results.txt"), "w") as f:
            f.write(str(results))
        if not args["fast_dev_run"]:
            wandb.log(results)

    print("Predictions complete")


if __name__ == "__main__":
    main()
