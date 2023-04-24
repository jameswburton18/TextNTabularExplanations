import numpy as np
import shap
import pickle
from datasets import load_dataset
from src.utils import format_text_pred
from transformers import pipeline, AutoTokenizer
import pandas as pd
from datasets import load_dataset, Dataset
from transformers.pipelines.pt_utils import KeyDataset
import re
import scipy as sp

# from src.models import Model
import lightgbm as lgb
from src.models import WeightedEnsemble, StackModel, AllAsTextModel, AllAsTextModel2
from src.joint_masker import JointMasker


def run_shap(model_type):
    # Data
    train_df = load_dataset(
        "james-burton/imdb_genre_prediction2", split="train"
    ).to_pandas()
    val_df = load_dataset(
        "james-burton/imdb_genre_prediction2", split="validation"
    ).to_pandas()
    test_df = load_dataset(
        "james-burton/imdb_genre_prediction2", split="test"
    ).to_pandas()
    tab_cols = [
        "Year",
        "Runtime (Minutes)",
        "Rating",
        "Votes",
        "Revenue (Millions)",
        "Metascore",
        "Rank",
    ]
    text_col = ["Description"]
    train_df_tab = train_df[tab_cols]
    val_df_tab = val_df[tab_cols]
    y_train = train_df["Genre_is_Drama"]
    y_val = val_df["Genre_is_Drama"]

    # Models
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    text_pipeline = pipeline(
        "text-classification",
        model="james-burton/imdb_genre_9",
        tokenizer=tokenizer,
        device="cuda:0",
    )
    tab_model = lgb.LGBMClassifier(random_state=42)
    tab_model.fit(train_df_tab, y_train)

    if model_type == "ensemble_50":
        model = WeightedEnsemble(
            tab_model=tab_model, text_pipeline=text_pipeline, text_weight=0.5
        )
    elif model_type == "ensemble_75":
        model = WeightedEnsemble(
            tab_model=tab_model, text_pipeline=text_pipeline, text_weight=0.75
        )
    elif model_type == "ensemble_25":
        model = WeightedEnsemble(
            tab_model=tab_model, text_pipeline=text_pipeline, text_weight=0.25
        )
    elif model_type == "stack":
        # Training set is the preditions from the tabular and text models
        text_val_preds = text_pipeline(list(val_df[text_col].values.squeeze()))
        text_val_preds = np.array([format_text_pred(pred) for pred in text_val_preds])
        tab_val_preds = tab_model.predict_proba(val_df_tab)

        # add text and tabular predictions to the val_df
        stack_val_df = val_df[tab_cols]
        stack_val_df["tab_preds"] = tab_val_preds[:, 1]
        stack_val_df["text_preds"] = text_val_preds[:, 1]

        stack_model = lgb.LGBMClassifier(random_state=42)
        stack_model.fit(stack_val_df, y_val)

        model = StackModel(
            tab_model=tab_model, text_pipeline=text_pipeline, stack_model=stack_model
        )
    elif model_type == "all_text":
        text_pipeline = pipeline(
            "text-classification",
            model="james-burton/imdb_genre_0",
            tokenizer=tokenizer,
            device="cuda:0",
        )
        model = AllAsTextModel(text_pipeline=text_pipeline)
    # elif model_type ==
    #     model = AllAsTextModel
    else:
        raise ValueError(f"Invalid model type of {model_type}")

    # We want to explain a single row
    np.random.seed(1)
    x = test_df[tab_cols + text_col].values  # .reshape(1, -1)
    # x = [7.7, 398972.0, 32.39, "offbeat romantic comedy"]

    masker = JointMasker(
        tab_df=train_df[tab_cols], tokenizer=tokenizer, collapse_mask_token=True
    )

    explainer = shap.explainers.Partition(model=model.predict, masker=masker)
    shap_vals = explainer(x)
    return shap_vals


def run_shap_multiple_text(model_type):
    # Data
    train_df = load_dataset(
        "james-burton/imdb_genre_prediction2", split="train"
    ).to_pandas()
    val_df = load_dataset(
        "james-burton/imdb_genre_prediction2", split="validation"
    ).to_pandas()
    test_df = load_dataset(
        "james-burton/imdb_genre_prediction2", split="test"
    ).to_pandas()
    tab_cols = [
        "Year",
        "Runtime (Minutes)",
        "Rating",
        "Votes",
        "Revenue (Millions)",
        "Metascore",
        "Rank",
    ]
    text_cols = ["Description", "Title"]
    train_df_tab = train_df[tab_cols]
    val_df_tab = val_df[tab_cols]
    y_train = train_df["Genre_is_Drama"]
    y_val = val_df["Genre_is_Drama"]

    # Models
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # elif model_type == "all_text":
    text_pipeline = pipeline(
        "text-classification",
        model="james-burton/imdb_genre_0",
        tokenizer=tokenizer,
        device="cuda:0",
    )
    model = AllAsTextModel2(text_pipeline=text_pipeline, cols=tab_cols + text_cols)

    np.random.seed(1)
    x = test_df[tab_cols + text_cols].values[:1]  # .reshape(1, -1)
    # x = [7.7, 398972.0, 32.39, "offbeat romantic comedy"]

    masker = JointMasker(
        tab_df=train_df[tab_cols],
        text_cols=text_cols,
        tokenizer=tokenizer,
        collapse_mask_token=True,
    )

    explainer = shap.explainers.Partition(model=model.predict, masker=masker)
    shap_vals = explainer(x)
    return shap_vals


if __name__ == "__main__":
    for model_type in [
        # "ensemble_50",
        # "ensemble_75",
        # "ensemble_25",
        # "stack",
        "all_text",
    ]:
        shap_vals = run_shap_multiple_text(model_type)
        # shap_vals = run_shap(model_type)
        # with open(f"shap_vals_{model_type}.pkl", "wb") as f:
        #     pickle.dump(shap_vals, f)

    print(shap_vals)
