import numpy as np
import shap
import pickle
from datasets import load_dataset
from src.utils import format_text_pred
from transformers import pipeline, AutoTokenizer
import pandas as pd
from datasets import load_dataset, Dataset
import os

# from src.models import Model
import lightgbm as lgb
from src.models import WeightedEnsemble, StackModel, AllAsTextModel
from src.joint_masker import JointMasker
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ds_type",
    type=str,
    default="wine",
    help="Name of dataset to use",
)
ds_type = parser.parse_args().ds_type

"""
prod_sent: 
* Tabular:  ["Product_Type"]
* Text: ["Product_Description"]

wine:
* Tabular: ["points", "price"]
* Text: ['country', 'description','province']

fake:
* Tabular: ['required_experience','required_education']
* Text: ["title", "description", "salary_range"]

kick: 
* Tabular: ['goal', 'disable_communication', 'country', 'currency', 'deadline', 'created_at']
* Text: ['name', 'desc', 'keywords']

jigsaw:
* Tabular: ['asian', 'atheist', 'bisexual', 'black', 'buddhist', 'christian', 'female', 'heterosexual', 'hindu', 'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability', 'jewish', 'latino', 'male', 'muslim', 'other_disability', 'other_gender', 'other_race_or_ethnicity', 'other_religion', 'other_sexual_orientation', 'physical_disability', 'psychiatric_or_mental_illness', 'transgender', 'white', 'funny', 'wow', 'sad', 'likes', 'disagree']
* Text: ['comment_text']
"""


def run_shap(model_type, ds_type, max_samples=100, test_set_size=100):
    output_root = "models/shap_vals/"

    if ds_type == "imdb_genre":
        ds_name = "james-burton/imdb_genre_prediction2"
        tab_cols = [
            "Year",
            "Runtime (Minutes)",
            "Rating",
            "Votes",
            "Revenue (Millions)",
            "Metascore",
            "Rank",
        ]
        text_cols = ["Description"]
        label_col = "Genre_is_Drama"
        if model_type == "all_text":
            text_model_name = "james-burton/imdb_genre_0"
        else:
            text_model_name = "james-burton/imdb_genre_9"

    elif ds_type == "prod_sent":
        ds_name = "james-burton/product_sentiment_machine_hack"
        tab_cols = ["Product_Type"]
        text_cols = ["Product_Description"]
        label_col = "Sentiment"
        if model_type == "all_text":
            text_model_name = "james-burton/prod_sent_0"
        else:
            text_model_name = "james-burton/prod_sent_9"
    elif ds_type == "fake":
        ds_name = "james-burton/fake_job_postings2"
        tab_cols = ["required_experience", "required_education"]
        text_cols = ["title", "description", "salary_range"]
        label_col = "fraudulent"
        if model_type == "all_text":
            text_model_name = "james-burton/fake_0"
        else:
            text_model_name = "james-burton/fake_9"
    elif ds_type == "kick":
        ds_name = "james-burton/kick_starter_funding"
        tab_cols = [
            "goal",
            "disable_communication",
            "country",
            "currency",
            "deadline",
            "created_at",
        ]
        text_cols = ["name", "desc", "keywords"]
        label_col = "final_status"
        if model_type == "all_text":
            text_model_name = "james-burton/kick_0"
        else:
            text_model_name = "james-burton/kick_9"
    elif ds_type == "jigsaw":
        ds_name = "james-burton/jigsaw_unintended_bias100K"
        tab_cols = [
            "asian",
            "atheist",
            "bisexual",
            "black",
            "buddhist",
            "christian",
            "female",
            "heterosexual",
            "hindu",
            "homosexual_gay_or_lesbian",
            "intellectual_or_learning_disability",
            "jewish",
            "latino",
            "male",
            "muslim",
            "other_disability",
            "other_gender",
            "other_race_or_ethnicity",
            "other_religion",
            "other_sexual_orientation",
            "physical_disability",
            "psychiatric_or_mental_illness",
            "transgender",
            "white",
            "funny",
            "wow",
            "sad",
            "likes",
            "disagree",
        ]
        text_cols = ["comment_text"]
        label_col = "target"
        if model_type == "all_text":
            text_model_name = "james-burton/jigsaw_0"
        else:
            text_model_name = "james-burton/jigsaw_9"
    elif ds_type == "wine":
        ds_name = "james-burton/wine_reviews"
        tab_cols = ["points", "price"]
        text_cols = ["country", "description", "province"]
        label_col = "variety"
        if model_type == "all_text":
            text_model_name = "james-burton/wine_0"
        else:
            text_model_name = "james-burton/wine_9"

    # Data
    train_df = load_dataset(ds_name, split="train").to_pandas()
    val_df = load_dataset(ds_name, split="validation").to_pandas()
    train_df_tab = train_df[tab_cols]
    val_df_tab = val_df[tab_cols]
    y_train = train_df[label_col]
    y_val = val_df[label_col]

    test_df = load_dataset(ds_name, split="test").to_pandas()
    test_df = test_df.sample(test_set_size, random_state=55)

    # Models
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    text_pipeline = pipeline(
        "text-classification",
        model=text_model_name,
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
        text_val_preds = text_pipeline(list(val_df[text_cols].values.squeeze()))
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
            model=text_model_name,
            tokenizer=tokenizer,
            device="cuda:0",
        )
        model = AllAsTextModel(text_pipeline=text_pipeline, cols=tab_cols + text_cols)
    else:
        raise ValueError(f"Invalid model type of {model_type}")

    np.random.seed(1)
    x = test_df[tab_cols + text_cols].values

    masker = JointMasker(
        tab_df=train_df[tab_cols],
        text_cols=text_cols,
        tokenizer=tokenizer,
        collapse_mask_token=True,
        max_samples=max_samples,
    )

    explainer = shap.explainers.Partition(model=model.predict, masker=masker)
    shap_vals = explainer(x)

    output_dir = os.path.join(output_root, ds_type)
    print(f"Results will be saved @: {output_dir}")

    # Make output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, f"shap_vals_{model_type}.pkl"), "wb") as f:
        pickle.dump(shap_vals, f)

    return shap_vals


# def run_shap_multiple_text(model_type, max_samples=100):
#     # Data
#     train_df = load_dataset(
#         "james-burton/imdb_genre_prediction2", split="train"
#     ).to_pandas()
#     val_df = load_dataset(
#         "james-burton/imdb_genre_prediction2", split="validation"
#     ).to_pandas()
#     test_df = load_dataset(
#         "james-burton/imdb_genre_prediction2", split="test"
#     ).to_pandas()
#     tab_cols = [
#         "Year",
#         "Runtime (Minutes)",
#         "Rating",
#         "Votes",
#         "Revenue (Millions)",
#         "Metascore",
#         "Rank",
#     ]
#     text_cols = ["Description"]  # , "Title"]
#     train_df_tab = train_df[tab_cols]
#     val_df_tab = val_df[tab_cols]
#     y_train = train_df["Genre_is_Drama"]
#     y_val = val_df["Genre_is_Drama"]

#     # Models
#     tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

#     # elif model_type == "all_text":
#     text_pipeline = pipeline(
#         "text-classification",
#         model="james-burton/imdb_genre_0",
#         tokenizer=tokenizer,
#         device="cuda:0",
#     )
#     model = AllAsTextModel(text_pipeline=text_pipeline, cols=tab_cols + text_cols)

#     np.random.seed(1)
#     x = test_df[tab_cols + text_cols].values[:1]  # .reshape(1, -1)
#     # x = [7.7, 398972.0, 32.39, "offbeat romantic comedy"]

#     masker = JointMasker(
#         tab_df=train_df[tab_cols],
#         text_cols=text_cols,
#         tokenizer=tokenizer,
#         collapse_mask_token=True,
#         max_samples=max_samples,
#     )

#     explainer = shap.explainers.Partition(model=model.predict, masker=masker)
#     shap_vals = explainer(x)

#     return shap_vals


if __name__ == "__main__":
    # for ds_type in ["imdb_genre", "prod_sent", "fake", "kick", "jigsaw", "wine"]:
    for model_type in [
        "ensemble_50",
        "ensemble_75",
        "ensemble_25",
        "stack",
        "all_text",
    ]:
        # shap_vals = run_shap_multiple_text(model_type)
        shap_vals = run_shap(model_type, ds_type=ds_type)

        # print(shap_vals)
