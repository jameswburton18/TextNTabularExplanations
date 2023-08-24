import numpy as np
import shap
import pickle
from datasets import load_dataset
from src.dataset_info import get_dataset_info
from transformers import pipeline, AutoTokenizer
import pandas as pd
from datasets import load_dataset, Dataset
import os
from tqdm import tqdm
from src.utils import token_segments, text_ft_index_ends

# from src.models import Model
import lightgbm as lgb
from src.models import WeightedEnsemble, StackModel, AllAsTextModel
from src.joint_masker import JointMasker
import argparse
import scipy as sp
from sklearn.metrics import roc_auc_score


def run_shap(
    model_type,
    ds_type,
    text_model_code,
    max_samples=100,
    test_set_size=100,
    tab_scale_factor=2,
):
    for text_model_code in [
        "disbert",
        "bert",
        "drob",
        "deberta",
    ]:
        for ds_type in ["prod_sent", "wine", "salary", "airbnb", "channel"]:
            for model_type in [
                "ensemble_25",
                "ensemble_50",
                "ensemble_75",
                "stack",
                "all_text",
                # "all_as_text_tnt_reorder",
                # "all_as_text_base_reorder",
            ]:
                di = get_dataset_info(ds_type, model_type)
                # Data
                train_df = load_dataset(
                    di.ds_name,
                    split="train",  # download_mode="force_redownload"
                ).to_pandas()
                y_train = train_df[di.label_col]

                test_df = load_dataset(
                    di.ds_name,
                    split="test",  # download_mode="force_redownload"
                ).to_pandas()
                # test_df = test_df.sample(test_set_size, random_state=55)

                if text_model_code == "disbert":
                    text_model_base = "distilbert-base-uncased"
                    my_text_model = di.text_model_name
                elif text_model_code == "bert":
                    text_model_base = "bert-base-uncased"
                    # 0s and 9s become 10s and 19s
                    my_text_model = (
                        di.text_model_name[:-1] + "1" + di.text_model_name[-1]
                    )
                elif text_model_code == "drob":
                    text_model_base = "distilroberta-base"
                    # 0s and 9s become 20s and 29s
                    my_text_model = (
                        di.text_model_name[:-1] + "2" + di.text_model_name[-1]
                    )
                elif text_model_code == "deberta":
                    text_model_base = "microsoft/deberta-v3-small"
                    # 0s and 9s become 30s and 39s
                    my_text_model = (
                        di.text_model_name[:-1] + "3" + di.text_model_name[-1]
                    )
                else:
                    raise ValueError(f"Invalid text model code of {text_model_code}")

                # Models
                tokenizer = AutoTokenizer.from_pretrained(
                    text_model_base, model_max_length=512
                )
                if model_type in [
                    "all_text",
                    # "all_as_text_tnt_reorder",
                    # "all_as_text_base_reorder",
                ]:
                    text_pipeline = pipeline(
                        "text-classification",
                        model=my_text_model,
                        tokenizer=tokenizer,
                        device="cuda:0",
                        truncation=True,
                        padding=True,
                        top_k=None,
                    )
                    # Define how to convert all columns to a single string
                    if model_type == "all_text":

                        def cols_to_str_fn(array):
                            return " | ".join(
                                [
                                    f"{col}: {val}"
                                    for col, val in zip(
                                        di.tab_cols + di.text_cols, array
                                    )
                                ]
                            )

                    # else:
                    #     # Reorder based on the new index order in di
                    #     def cols_to_str_fn(array): return " | ".join(
                    #         [
                    #             f"{col}: {val}"
                    #             for _, col, val in sorted(
                    #                 zip(di.new_idx_order, di.tab_cols + di.text_cols, array)
                    #             )
                    #         ]
                    #     )

                    model = AllAsTextModel(
                        text_pipeline=text_pipeline,
                        cols_to_str_fn=cols_to_str_fn,
                        # cols=di.tab_cols + di.text_cols
                    )
                else:
                    text_pipeline = pipeline(
                        "text-classification",
                        model=my_text_model,
                        tokenizer=tokenizer,
                        device="cuda:0",
                        truncation=True,
                        padding=True,
                        top_k=None,
                    )
                    # Define how to convert the text columns to a single string
                    if len(di.text_cols) == 1:

                        def cols_to_str_fn(array):
                            return array[0]

                    else:

                        def cols_to_str_fn(array):
                            return " | ".join(
                                [
                                    f"{col}: {val}"
                                    for col, val in zip(di.text_cols, array)
                                ]
                            )

                    # LightGBM requires explicitly marking categorical features
                    train_df[di.categorical_cols] = train_df[
                        di.categorical_cols
                    ].astype("category")
                    test_df[di.categorical_cols] = test_df[di.categorical_cols].astype(
                        "category"
                    )

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
                            di.ds_name,
                            split="validation",
                            download_mode="force_redownload",
                        ).to_pandas()
                        val_df[di.categorical_cols] = val_df[
                            di.categorical_cols
                        ].astype("category")
                        y_val = val_df[di.label_col]
                        val_text = list(
                            map(cols_to_str_fn, val_df[di.text_cols].values)
                        )

                        # Training set is the preditions from the tabular and text models on the validation set
                        # plus the tabular features from the validation set
                        text_val_preds = text_pipeline(val_text)
                        # text_val_preds = np.array(
                        #     [format_text_pred(pred) for pred in text_val_preds]
                        # )
                        text_val_preds = np.array(
                            [[lab["score"] for lab in pred] for pred in text_val_preds]
                        )

                        # add text and tabular predictions to the val_df
                        stack_val_df = val_df[di.tab_cols]
                        tab_val_preds = tab_model.predict_proba(stack_val_df)
                        for i in range(text_val_preds.shape[1]):
                            stack_val_df[f"text_pred_{i}"] = text_val_preds[:, i]
                        for i in range(tab_val_preds.shape[1]):
                            stack_val_df[f"tab_pred_{i}"] = tab_val_preds[:, i]

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
        # test_sample_vals = test_df_sample[di.tab_cols + di.text_cols].values
        test_vals = test_df[di.tab_cols + di.text_cols].values

        preds = model.predict(test_vals)
        actual = test_df[di.label_col].values
        return (
            # model.predict(test_sample_vals),
            # test_df_sample[di.label_col].values,
            "nan",
            "nan",
            preds,
            actual,
        )
