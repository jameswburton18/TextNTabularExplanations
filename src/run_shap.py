import numpy as np
import shap
import pickle
from datasets import load_dataset
from src.utils import format_text_pred, get_dataset_info
from transformers import pipeline, AutoTokenizer
import pandas as pd
from datasets import load_dataset, Dataset
import os

# from src.models import Model
import lightgbm as lgb
from src.models import WeightedEnsemble, StackModel, AllAsTextModel
from src.joint_masker import JointMasker
import argparse
import scipy as sp

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ds_type",
    type=str,
    default="wine",
    help="Name of dataset to use",
)


def run_shap(model_type, ds_type, max_samples=100, test_set_size=100):
    di = get_dataset_info(ds_type, model_type)
    # Data
    train_df = load_dataset(
        di.ds_name, split="train", download_mode="force_redownload"
    ).to_pandas()
    y_train = train_df[di.label_col]

    test_df = load_dataset(
        di.ds_name, split="test", download_mode="force_redownload"
    ).to_pandas()
    test_df = test_df.sample(test_set_size, random_state=55)

    # Models
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    if model_type == "all_text":
        text_pipeline = pipeline(
            "text-classification",
            model=di.text_model_name,
            tokenizer=tokenizer,
            device="cuda:0",
            truncation=True,
            padding=True,
            return_all_scores=True,
        )
        # Define how to convert all columns to a single string
        cols_to_str_fn = lambda array: " | ".join(
            [f"{col}: {val}" for col, val in zip(di.tab_cols + di.text_cols, array)]
        )
        model = AllAsTextModel(
            text_pipeline=text_pipeline, cols=di.tab_cols + di.text_cols
        )
    else:
        text_pipeline = pipeline(
            "text-classification",
            model=di.text_model_name,
            tokenizer=tokenizer,
            device="cuda:0",
            truncation=True,
            padding=True,
            return_all_scores=True,
        )
        # Define how to convert the text columns to a single string
        if len(di.text_cols) == 1:
            cols_to_str_fn = lambda array: array[0]
        else:
            cols_to_str_fn = lambda array: " | ".join(
                [f"{col}: {val}" for col, val in zip(di.text_cols, array)]
            )

        # LightGBM requires explicitly marking categorical features
        train_df[di.categorical_cols] = train_df[di.categorical_cols].astype("category")
        test_df[di.categorical_cols] = test_df[di.categorical_cols].astype("category")

        tab_model = lgb.LGBMClassifier(random_state=42)
        tab_model.fit(train_df[di.tab_cols], y_train)

        if model_type == "ensemble_50":
            model = WeightedEnsemble(
                tab_model=tab_model,
                text_pipeline=text_pipeline,
                text_weight=0.5,
                cols_to_str_fn=cols_to_str_fn,
            )
        elif model_type == "ensemble_75":
            model = WeightedEnsemble(
                tab_model=tab_model,
                text_pipeline=text_pipeline,
                text_weight=0.75,
                cols_to_str_fn=cols_to_str_fn,
            )
        elif model_type == "ensemble_25":
            model = WeightedEnsemble(
                tab_model=tab_model,
                text_pipeline=text_pipeline,
                text_weight=0.25,
                cols_to_str_fn=cols_to_str_fn,
            )
        elif model_type == "stack":
            """
            For the stack model, we make predictions on the validation set. These predictions
            are then used as features for the stack model (another LightGBM model) along with
            the other tabular features. In doing so the stack model learns, depending on the
            tabular features, when to trust the tabular model and when to trust the text model.
            """
            val_df = load_dataset(
                di.ds_name, split="validation", download_mode="force_redownload"
            ).to_pandas()
            val_df[di.categorical_cols] = val_df[di.categorical_cols].astype("category")
            y_val = val_df[di.label_col]
            val_text = list(map(cols_to_str_fn, val_df[di.text_cols].values))

            # Training set is the preditions from the tabular and text models on the validation set
            # plus the tabular features from the validation set
            text_val_preds = text_pipeline(val_text)
            text_val_preds = np.array(
                [format_text_pred(pred) for pred in text_val_preds]
            )

            # add text and tabular predictions to the val_df
            stack_val_df = val_df[di.tab_cols]
            tab_val_preds = tab_model.predict_proba(stack_val_df)
            stack_val_df["tab_preds"] = tab_val_preds[:, 1]
            stack_val_df["text_preds"] = text_val_preds[:, 1]

            stack_model = lgb.LGBMClassifier(random_state=42)
            stack_model.fit(stack_val_df, y_val)

            model = StackModel(
                tab_model=tab_model,
                text_pipeline=text_pipeline,
                stack_model=stack_model,
                cols_to_str_fn=cols_to_str_fn,
            )
        else:
            raise ValueError(f"Invalid model type of {model_type}")

    np.random.seed(1)
    x = test_df[di.tab_cols + di.text_cols].values
    x = np.array([[85, 25.0, "US", "tough and chewy", "Oregon"]], dtype=object)
    # x = np.array([[9.0, "Hello world"]], dtype=object)
    # x = np.array(
    #     [[5.0, 1.0, "Senior Strategist", "Cutting Edge", "None"]], dtype=object
    # )

    # We need to load the ordinal dataset so that we can calculate the correlations for the masker
    ord_ds_name = get_dataset_info(ds_type, "ordinal").ds_name
    ord_train_df = load_dataset(ord_ds_name, split="train").to_pandas()

    # Clustering only valid if there is more than one column
    if len(di.tab_cols) > 1:
        tab_pt = sp.cluster.hierarchy.complete(
            sp.spatial.distance.pdist(
                ord_train_df[di.tab_cols]
                .fillna(ord_train_df[di.tab_cols].median())
                .values.T,
                metric="correlation",
            )
        )
    else:
        tab_pt = None

    masker = JointMasker(
        tab_df=train_df[di.tab_cols],
        text_cols=di.text_cols,
        cols_to_str_fn=cols_to_str_fn,
        tokenizer=tokenizer,
        collapse_mask_token=True,
        max_samples=max_samples,
        tab_partition_tree=tab_pt,
    )

    explainer = shap.explainers.Partition(model=model.predict, masker=masker)
    shap_vals = explainer(x)

    output_dir = os.path.join("models/shap_vals/", ds_type)
    print(f"Results will be saved @: {output_dir}")

    # Make output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, f"shap_vals_{model_type}.pkl"), "wb") as f:
        pickle.dump(shap_vals, f)

    return shap_vals


if __name__ == "__main__":
    ds_type = parser.parse_args().ds_type
    for model_type in [
        "ensemble_50",
        # "ensemble_75",
        # "ensemble_25",
        # "stack",
        # "all_text",
    ]:
        # shap_vals = run_shap_multiple_text(model_type)
        shap_vals = run_shap(model_type, ds_type=ds_type)

        # print(shap_vals)
