from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_from_disk
import wandb
import os
import yaml
import argparse
from transformers.trainer_callback import EarlyStoppingCallback
import evaluate
import numpy as np
from lion_pytorch import Lion

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="testing",
    help="Name of config from the the multi_config.yaml file",
)
config_type = parser.parse_args().config


def main():
    # Import yaml file
    with open("configs/train_default.yaml") as f:
        args = yaml.safe_load(f)

    # Update default args with chosen config
    if config_type != "default":
        with open("configs/train_configs.yaml") as f:
            yaml_configs = yaml.safe_load_all(f)
            yaml_args = next(
                conf for conf in yaml_configs if conf["config"] == config_type
            )
        args.update(yaml_args)
        print(f"Updating with:\n{yaml_args}\n")
    print(f"\n{args}\n")

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        args["model_base"], num_labels=1, problem_type="regression"
    )
    tokenizer = AutoTokenizer.from_pretrained(args["model_base"])

    # Dataset
    dataset = load_from_disk("data/processed/airbnb/summaries")
    
    # Tokenize the dataset
    def encode(examples):
        return {"label": np.array([examples["label"]]),
                **tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)}
    dataset = dataset.map(encode,load_from_cache_file=True)

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
            project="Airbnb",
            tags=args["tags"],
            save_code=True,
            config={"my_args/" + k: v for k, v in args.items()},
        )
        os.environ["WANDB_LOG_MODEL"] = "True"
        output_dir = os.path.join(args["output_root"], wandb.run.name)
        print(f"Results will be saved @: {output_dir}")

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
        remove_unused_columns=False,
        # # PyTorch 2.0
        # torch_compile=True,
    )

    mse_metric = evaluate.load("mse")
    if args["lion_optim"]:
        opt = Lion(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
        sched = None
    
    trainer = Trainer(
        model=model,
        optimizers=(opt, sched) if args['lion_optim'] else (None, None),
        args=training_args,
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
        print("Training complete")
    # Predict on the test set
    if args["do_predict"]:
        print("***** Running Prediction *****")
        # Test the model
        results = trainer.evaluate(dataset["test"], metric_key_prefix="test")
        results['test_rmse'] = np.sqrt(results['test_loss'])

        # Save the predictions
        with open(os.path.join(output_dir, "test_results.txt"), "w") as f:
            f.write(str(results))
        if not args["fast_dev_run"]:
            wandb.log(results)

    print("Predictions complete")


if __name__ == "__main__":
    main()

