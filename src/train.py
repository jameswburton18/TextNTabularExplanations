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

    # Dataset
    ds_type = args['dataset']
    if ds_type == 'airbnb':
        ds_name = 'james-burton/airbnb_summaries'
        project = 'Airbnb'
        label_col, text_col = 'scaled_label', 'text'
        prob_type, num_labels = 'regression', 1
    elif ds_type == 'books':
        ds_name = 'james-burton/books_price_prediction'
        project = 'Books'
        label_col, text_col = 'scaled_label', 'text'
        prob_type, num_labels = 'regression', 1
    elif ds_type == 'imdb':
        ds_name = 'james-burton/imdb_gross'
        project = 'IMDB'
        label_col, text_col = 'scaled_label', 'text'
        prob_type, num_labels = 'regression', 1
    elif ds_type == 'imdb_genre':
        ds_name = 'james-burton/imdb_genre_prediction2'
        project = 'IMDB Genre'
        label_col, text_col = 'Genre_is_Drama', 'Description'
        prob_type, num_labels = 'single_label_classification', 2
    elif ds_type == 'imdb_genre_text':
        ds_name = 'james-burton/imdb_genre_prediction_all_text'
        project = 'IMDB Genre All Text'
        label_col, text_col = 'Genre_is_Drama', 'text'
        prob_type, num_labels = 'single_label_classification', 2
    dataset = load_dataset(ds_name)
    if prob_type == 'regression':
        mean_price = np.mean(dataset['train']['label'])
        std_price = np.std(dataset['train']['label'])
    
        # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        args["model_base"], num_labels=num_labels, problem_type=prob_type
    )
    tokenizer = AutoTokenizer.from_pretrained(args["model_base"])
    
    # Tokenize the dataset
    def encode(examples, ds_type=ds_type):
        return {
            "labels": np.array([examples[label_col]]),
                    **tokenizer(examples[text_col], truncation=True, padding="max_length")}
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
            project=project,
            tags=args["tags"],
            save_code=True,
            config={"my_args/" + k: v for k, v in args.items()},
        )
        os.environ["WANDB_LOG_MODEL"] = "True"
        output_dir = os.path.join(args["output_root"],ds_type, wandb.run.name)
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
        torch_compile=args['pytorch2.0'], # Needs to be true if PyTorch 2.0
    )

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
        preds = trainer.predict(dataset["test"]).predictions
        labels = [l[0] for l in dataset['test']['labels']]
        if prob_type == 'regression':
            unscaled_preds = preds * std_price + mean_price
            unscaled_refs = [[x[0]*std_price + mean_price] for x in dataset['test']['label']]
            results['test_unscaled_rmse'] = np.sqrt(np.mean((unscaled_preds - unscaled_refs)**2))
            results['test_rmse'] = np.sqrt(results['test_loss'])
        else:
            results['test/accuracy'] = np.mean(np.argmax(preds, axis=1) == labels)
            results['test/precision'] = precision_score(labels, np.argmax(preds, axis=1))
            results['test/recall'] = recall_score(labels, np.argmax(preds, axis=1))
            results['test/roc_auc'] = roc_auc_score(labels, preds[:,1])

        # Save the predictions
        with open(os.path.join(output_dir, "test_results.txt"), "w") as f:
            f.write(str(results))
        if not args["fast_dev_run"]:
            wandb.log(results)

    print("Predictions complete")


if __name__ == "__main__":
    main()

