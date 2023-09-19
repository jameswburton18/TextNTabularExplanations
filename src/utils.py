import numpy as np
from shap.utils import safe_isinstance
from shap.utils.transformers import (
    SENTENCEPIECE_TOKENIZERS,
    getattr_silent,
)
from dataclasses import dataclass
from typing import List
import yaml
from warnings import warn
import yaml


class ConfigLoader:
    def __init__(self, config_name, configs_path, default_path=None):
        if default_path is not None:
            with open(default_path) as f:
                args = yaml.safe_load(f)

        # Update default args with chosen config
        if config_name != "default":
            with open(configs_path) as f:
                yaml_configs = yaml.safe_load_all(f)
                try:
                    yaml_args = next(
                        conf for conf in yaml_configs if conf["config"] == config_name
                    )
                except StopIteration:
                    raise ValueError(
                        f"Config name {config_name} not found in {configs_path}"
                    )
            if default_path is not None:
                args.update(yaml_args)
                print(f"Updating with:\n{yaml_args}\n")
            else:
                args = yaml_args
        print(f"\n{args}\n")
        for key, value in args.items():
            setattr(self, key, value)


def row_to_string(row, cols):
    row["text"] = " | ".join(f"{col}: {row[col]}" for col in cols)
    return row


def multiple_row_to_string(row, cols, multiplier=1, nodesc=False):
    row["text"] = " | ".join(
        f'{col}: {(str(row[col]) + " ")*multiplier}' for col in cols
    )
    if not nodesc:
        row["text"] = row["text"] + " | Description: " + row["Description"]
    return row


def prepare_text(dataset, version, di, reverse=False, model_name=None):
    """This is all for preparing the text part of the dataset
    Could be made more robust by referring to dataset_info.py instead"""

    # Used for reorder versions
    if model_name == "bert-base-uncased":
        model_code = "bert"
    elif model_name == "distilbert-base-uncased":
        model_code = "disbert"
    elif model_name == "distilroberta-base":
        model_code = "drob"
    elif model_name == "microsoft/deberta-v3-small":
        model_code = "deberta"
    else:
        model_code = None

    if version == "all_as_text":
        cols = di.tab_cols + di.text_cols
        dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
        return dataset
    elif version == "text_col_only":
        if len(di.text_cols) == 1:
            # dataset rename column
            dataset = dataset.rename_column(di.text_cols[0], "text")
        else:
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": di.text_cols})
    elif version == "all_as_text_base_reorder":
        cols = di.base_reorder_cols[model_code]
        cols = cols[::-1] if reverse else cols
        dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
        return dataset
    elif version == "all_as_text_tnt_reorder":
        cols = di.tnt_reorder_cols[model_code]
        cols = cols[::-1] if reverse else cols
        dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
        return dataset

    else:
        raise ValueError(
            f"Unknown dataset type ({di.config}) and version ({version}) combination"
        )


def format_text_pred(pred):
    scores = [p["score"] for p in pred]
    order = [int(p["label"][6:]) for p in pred]
    return np.array(
        [scores[i] for i in sorted(range(len(scores)), key=lambda x: order[x])]
    )


def format_fts_for_plotting(fts, tab_data):
    for i in range(len(tab_data)):
        fts[i] = fts[i] + f" = {tab_data[i]}   "
    # for j in range(len(tab_data), len(fts)):
    #     fts[j] = fts[j] + ""
    return fts


def text_ft_index_ends(text_fts, tokenizer):
    lens = []
    sent_indices = []
    for idx, col in enumerate(text_fts):
        # First text col
        if lens == []:
            tokens, token_ids = token_segments(str(col), tokenizer)
            # -1 as we don't use SEP tokens (unless it's the only text col)
            also_last = 1 if len(text_fts) == 1 else 0
            token_len = len(tokens) - 1 + also_last
            lens.append(token_len - 1)
            sent_indices.extend([idx] * token_len)
        # Last text col
        elif idx == len(text_fts) - 1:
            tokens, token_ids = token_segments(str(col), tokenizer)
            # -1 for CLS tokens
            token_len = len(tokens) - 1
            lens.append(lens[-1] + token_len)
            sent_indices.extend([idx] * token_len)
        # Middle text cols
        else:
            tokens, token_ids = token_segments(str(col), tokenizer)
            # -2 for CLS and SEP tokens
            token_len = len(tokens) - 2
            lens.append(lens[-1] + token_len)
            sent_indices.extend([idx] * token_len)

    return lens[:-1]


def token_segments(s, tokenizer):
    """Same as Text masker"""
    """ Returns the substrings associated with each token in the given string.
    """

    try:
        token_data = tokenizer(s, return_offsets_mapping=True)
        offsets = token_data["offset_mapping"]
        offsets = [(0, 0) if o is None else o for o in offsets]
        parts = [
            s[offsets[i][0] : max(offsets[i][1], offsets[i + 1][0])]
            for i in range(len(offsets) - 1)
        ]
        parts.append(s[offsets[len(offsets) - 1][0] : offsets[len(offsets) - 1][1]])
        return parts, token_data["input_ids"]
    except (
        NotImplementedError,
        TypeError,
    ):  # catch lack of support for return_offsets_mapping
        token_ids = tokenizer(s)["input_ids"]
        if hasattr(tokenizer, "convert_ids_to_tokens"):
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
        else:
            tokens = [tokenizer.decode([id]) for id in token_ids]
        if hasattr(tokenizer, "get_special_tokens_mask"):
            special_tokens_mask = tokenizer.get_special_tokens_mask(
                token_ids, already_has_special_tokens=True
            )
            # avoid masking separator tokens, but still mask beginning of sentence and end of sentence tokens
            special_keep = [
                getattr_silent(tokenizer, "sep_token"),
                getattr_silent(tokenizer, "mask_token"),
            ]
            for i, v in enumerate(special_tokens_mask):
                if v == 1 and (
                    tokens[i] not in special_keep or i + 1 == len(special_tokens_mask)
                ):
                    tokens[i] = ""

        # add spaces to separate the tokens (since we want segments not tokens)
        if safe_isinstance(tokenizer, SENTENCEPIECE_TOKENIZERS):
            for i, v in enumerate(tokens):
                if v.startswith("_"):
                    tokens[i] = " " + tokens[i][1:]
        else:
            for i, v in enumerate(tokens):
                if v.startswith("##"):
                    tokens[i] = tokens[i][2:]
                elif v != "" and i != 0:
                    tokens[i] = " " + tokens[i]

        return tokens, token_ids


###################
# Legacy
@dataclass
class DatasetInfo:
    """Container for dataset information."""

    ds_name: str
    tab_cols: List[str]
    categorical_cols: List[str]
    text_cols: List[str]
    label_names: List[str]
    label_col: str
    num_labels: int
    prob_type: str
    wandb_proj_name: str
    tnt_reorder_cols: List[str] = None
    base_reorder_cols: List[str] = None
    text_model_name: str = None


def legacy_get_dataset_info(ds_type, model_type=None):
    if ds_type in ["imdb_genre", "imdb_genre_prediction"]:
        match model_type:
            case None:
                text_model_name = None
                ds_name = None
                warn(
                    f"No model type specified for {ds_type}. (This is fine during dataset creation)"
                )
            case "all_text" | "all_as_text":
                print(f"Using dataset {ds_type}, all as text version")
                text_model_name = "james-burton/imdb_genre_0"
                ds_name = "james-burton/imdb_genre_prediction_all_text"
            case _:
                print(f"Using dataset {ds_type}, ordinal version")
                text_model_name = "james-burton/imdb_genre_9"
                ds_name = "james-burton/imdb_genre_prediction_ordinal"
        return DatasetInfo(
            ds_name=ds_name,
            tab_cols=[
                "Year",
                "Runtime (Minutes)",
                "Rating",
                "Votes",
                "Revenue (Millions)",
                "Metascore",
                "Rank",
            ],
            categorical_cols=[],
            text_cols=["Description"],
            label_col="Genre_is_Drama",
            num_labels=2,
            prob_type="single_label_classification",
            wandb_proj_name="IMDB Genre",
            text_model_name=text_model_name,
            label_names=["False", "True"],
            tnt_reorder_cols={
                "bert": [
                    "Description",
                    "Rank",
                    "Votes",
                    "Revenue (Millions)",
                    "Metascore",
                    "Runtime (Minutes)",
                    "Rating",
                    "Year",
                ],
                "disbert": [
                    "Description",
                    "Votes",
                    "Rank",
                    "Revenue (Millions)",
                    "Runtime (Minutes)",
                    "Year",
                    "Metascore",
                    "Rating",
                ],
                "drob": [
                    "Description",
                    "Year",
                    "Rank",
                    "Revenue (Millions)",
                    "Votes",
                    "Metascore",
                    "Runtime (Minutes)",
                    "Rating",
                ],
                "deberta": [
                    "Description",
                    "Rank",
                    "Revenue (Millions)",
                    "Votes",
                    "Runtime (Minutes)",
                    "Metascore",
                    "Rating",
                    "Year",
                ],
            },
            base_reorder_cols={
                "bert": [
                    "Description",
                    "Metascore",
                    "Revenue (Millions)",
                    "Rank",
                    "Rating",
                    "Votes",
                    "Runtime (Minutes)",
                    "Year",
                ],
                "disbert": [
                    "Description",
                    "Metascore",
                    "Rank",
                    "Rating",
                    "Revenue (Millions)",
                    "Votes",
                    "Year",
                    "Runtime (Minutes)",
                ],
                "drob": [
                    "Description",
                    "Revenue (Millions)",
                    "Metascore",
                    "Votes",
                    "Year",
                    "Rank",
                    "Runtime (Minutes)",
                    "Rating",
                ],
                "deberta": [
                    "Description",
                    "Rating",
                    "Metascore",
                    "Rank",
                    "Revenue (Millions)",
                    "Year",
                    "Votes",
                    "Runtime (Minutes)",
                ],
            },
        )
    elif ds_type in ["prod_sent", "product_sentiment_machine_hack"]:
        match model_type:
            case None:
                text_model_name = None
                ds_name = None
                warn(
                    f"No model type specified for {ds_type}. (This is fine during dataset creation)"
                )
            case "all_text" | "all_as_text":
                print(f"Using dataset {ds_type}, all as text version")
                text_model_name = "james-burton/prod_sent_0"
                ds_name = "james-burton/product_sentiment_machine_hack_all_text"
            case _:
                print(f"Using dataset {ds_type}, ordinal version")
                text_model_name = "james-burton/prod_sent_9"
                ds_name = "james-burton/product_sentiment_machine_hack_ordinal"
        return DatasetInfo(
            ds_name=ds_name,
            tab_cols=["Product_Type"],
            categorical_cols=["Product_Type"],
            text_cols=["Product_Description"],
            label_col="Sentiment",
            num_labels=4,
            prob_type="single_label_classification",
            wandb_proj_name="Product Sentiment",
            text_model_name=text_model_name,
            label_names=["0", "1", "2", "3"],
            base_reorder_cols={
                "bert": ["Product_Type", "Product_Description"],
                "disbert": ["Product_Type", "Product_Description"],
                "drob": ["Product_Description", "Product_Type"],
                "deberta": ["Product_Type", "Product_Description"],
            },
            tnt_reorder_cols={
                "bert": ["Product_Type", "Product_Description"],
                "disbert": ["Product_Type", "Product_Description"],
                "drob": ["Product_Type", "Product_Description"],
                "deberta": ["Product_Type", "Product_Description"],
            },
        )
    elif ds_type in ["fake", "fake_job_postings2"]:
        match model_type:
            case None:
                text_model_name = None
                ds_name = None
                warn(
                    f"No model type specified for {ds_type}. (This is fine during dataset creation)"
                )
            case "all_text" | "all_as_text":
                print(f"Using dataset {ds_type}, all as text version")
                text_model_name = "james-burton/fake_0"
                ds_name = "james-burton/fake_job_postings2_all_text"
            case _:
                print(f"Using dataset {ds_type}, ordinal version")
                text_model_name = "james-burton/fake_9"
                ds_name = "james-burton/fake_job_postings2_ordinal"
        return DatasetInfo(
            ds_name=ds_name,
            tab_cols=["required_experience", "required_education"],
            categorical_cols=["required_experience", "required_education"],
            text_cols=["title", "description", "salary_range"],
            label_col="fraudulent",
            num_labels=2,
            prob_type="single_label_classification",
            wandb_proj_name="Fake Job Postings",
            text_model_name=text_model_name,
            label_names=["0", "1"],
            tnt_reorder_cols={
                "bert": [
                    "description",
                    "title",
                    "required_experience",
                    "required_education",
                    "salary_range",
                ],
                "disbert": [
                    "description",
                    "title",
                    "required_education",
                    "required_experience",
                    "salary_range",
                ],
                "drob": [
                    "description",
                    "title",
                    "required_education",
                    "required_experience",
                    "salary_range",
                ],
                "deberta": [
                    "description",
                    "title",
                    "required_education",
                    "required_experience",
                    "salary_range",
                ],
            },
            base_reorder_cols={
                "bert": [
                    "description",
                    "title",
                    "required_education",
                    "required_experience",
                    "salary_range",
                ],
                "disbert": [
                    "description",
                    "title",
                    "required_education",
                    "required_experience",
                    "salary_range",
                ],
                "drob": [
                    "description",
                    "title",
                    "required_experience",
                    "required_education",
                    "salary_range",
                ],
                "deberta": [
                    "description",
                    "title",
                    "required_education",
                    "salary_range",
                    "required_experience",
                ],
            },
        )
    elif ds_type in ["kick", "kick_starter_funding"]:
        match model_type:
            case None:
                text_model_name = None
                ds_name = None
                warn(
                    f"No model type specified for {ds_type}. (This is fine during dataset creation)"
                )
            case "all_text" | "all_as_text":
                print(f"Using dataset {ds_type}, all as text version")
                text_model_name = "james-burton/kick_0"
                ds_name = "james-burton/kick_starter_funding_all_text"
            case _:
                print(f"Using dataset {ds_type}, ordinal version")
                text_model_name = "james-burton/kick_9"
                ds_name = "james-burton/kick_starter_funding_ordinal"
        return DatasetInfo(
            ds_name=ds_name,
            tab_cols=[
                "goal",
                "disable_communication",
                "country",
                "currency",
                "deadline",
                "created_at",
            ],
            categorical_cols=["disable_communication", "country", "currency"],
            text_cols=["name", "desc", "keywords"],
            label_col="final_status",
            num_labels=2,
            prob_type="single_label_classification",
            wandb_proj_name="Kickstarter",
            text_model_name=text_model_name,
            label_names=["0", "1"],
            tnt_reorder_cols={
                "bert": [
                    "desc",
                    "goal",
                    "name",
                    "created_at",
                    "keywords",
                    "deadline",
                    "currency",
                    "country",
                    "disable_communication",
                ],
                "disbert": [
                    "desc",
                    "goal",
                    "name",
                    "keywords",
                    "created_at",
                    "deadline",
                    "currency",
                    "country",
                    "disable_communication",
                ],
                "drob": [
                    "desc",
                    "goal",
                    "name",
                    "keywords",
                    "deadline",
                    "created_at",
                    "currency",
                    "country",
                    "disable_communication",
                ],
                "deberta": [
                    "desc",
                    "goal",
                    "name",
                    "deadline",
                    "keywords",
                    "created_at",
                    "currency",
                    "country",
                    "disable_communication",
                ],
            },
            base_reorder_cols={
                "bert": [
                    "desc",
                    "goal",
                    "name",
                    "keywords",
                    "created_at",
                    "deadline",
                    "country",
                    "disable_communication",
                    "currency",
                ],
                "disbert": [
                    "desc",
                    "goal",
                    "name",
                    "created_at",
                    "keywords",
                    "deadline",
                    "currency",
                    "disable_communication",
                    "country",
                ],
                "drob": [
                    "desc",
                    "name",
                    "goal",
                    "keywords",
                    "created_at",
                    "deadline",
                    "country",
                    "currency",
                    "disable_communication",
                ],
                "deberta": [
                    "goal",
                    "desc",
                    "name",
                    "keywords",
                    "deadline",
                    "created_at",
                    "currency",
                    "country",
                    "disable_communication",
                ],
            },
        )
    elif ds_type in ["jigsaw", "jigsaw_unintended_bias100K"]:
        match model_type:
            case None:
                text_model_name = None
                ds_name = None
                warn(
                    f"No model type specified for {ds_type}. (This is fine during dataset creation)"
                )
            case "all_text" | "all_as_text":
                print(f"Using dataset {ds_type}, all as text version")
                text_model_name = "james-burton/jigsaw_0"
                ds_name = "james-burton/jigsaw_unintended_bias100K_all_text"
            case _:
                print(f"Using dataset {ds_type}, ordinal version")
                text_model_name = "james-burton/jigsaw_9"
                ds_name = "james-burton/jigsaw_unintended_bias100K_ordinal"
        return DatasetInfo(
            ds_name=ds_name,
            tab_cols=[
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
            ],
            categorical_cols=[],
            text_cols=["comment_text"],
            label_col="target",
            num_labels=2,
            prob_type="single_label_classification",
            wandb_proj_name="Jigsaw",
            text_model_name=text_model_name,
            label_names=["False", "True"],
            tnt_reorder_cols={
                "bert": [
                    "comment_text",
                    "atheist",
                    "other_gender",
                    "buddhist",
                    "hindu",
                    "muslim",
                    "jewish",
                    "other_religion",
                    "christian",
                    "latino",
                    "black",
                    "white",
                    "other_race_or_ethnicity",
                    "asian",
                    "female",
                    "male",
                    "intellectual_or_learning_disability",
                    "other_disability",
                    "psychiatric_or_mental_illness",
                    "physical_disability",
                    "heterosexual",
                    "other_sexual_orientation",
                    "homosexual_gay_or_lesbian",
                    "bisexual",
                    "transgender",
                    "sad",
                    "disagree",
                    "likes",
                    "funny",
                    "wow",
                ],
                "disbert": [
                    "comment_text",
                    "atheist",
                    "other_gender",
                    "buddhist",
                    "hindu",
                    "christian",
                    "other_religion",
                    "jewish",
                    "muslim",
                    "latino",
                    "black",
                    "heterosexual",
                    "male",
                    "female",
                    "other_sexual_orientation",
                    "other_race_or_ethnicity",
                    "white",
                    "asian",
                    "psychiatric_or_mental_illness",
                    "physical_disability",
                    "other_disability",
                    "intellectual_or_learning_disability",
                    "transgender",
                    "homosexual_gay_or_lesbian",
                    "bisexual",
                    "likes",
                    "disagree",
                    "sad",
                    "funny",
                    "wow",
                ],
                "drob": [
                    "comment_text",
                    "atheist",
                    "psychiatric_or_mental_illness",
                    "intellectual_or_learning_disability",
                    "other_disability",
                    "physical_disability",
                    "female",
                    "male",
                    "latino",
                    "other_race_or_ethnicity",
                    "asian",
                    "white",
                    "black",
                    "jewish",
                    "muslim",
                    "christian",
                    "other_religion",
                    "homosexual_gay_or_lesbian",
                    "bisexual",
                    "transgender",
                    "heterosexual",
                    "other_sexual_orientation",
                    "other_gender",
                    "hindu",
                    "buddhist",
                    "sad",
                    "disagree",
                    "likes",
                    "funny",
                    "wow",
                ],
                "deberta": [
                    "comment_text",
                    "likes",
                    "disagree",
                    "psychiatric_or_mental_illness",
                    "physical_disability",
                    "white",
                    "intellectual_or_learning_disability",
                    "black",
                    "male",
                    "other_disability",
                    "female",
                    "latino",
                    "asian",
                    "transgender",
                    "other_race_or_ethnicity",
                    "other_sexual_orientation",
                    "homosexual_gay_or_lesbian",
                    "bisexual",
                    "heterosexual",
                    "other_gender",
                    "atheist",
                    "muslim",
                    "jewish",
                    "other_religion",
                    "christian",
                    "hindu",
                    "buddhist",
                    "sad",
                    "funny",
                    "wow",
                ],
            },
            base_reorder_cols={
                "bert": [
                    "comment_text",
                    "other_sexual_orientation",
                    "christian",
                    "hindu",
                    "female",
                    "heterosexual",
                    "physical_disability",
                    "buddhist",
                    "other_religion",
                    "homosexual_gay_or_lesbian",
                    "other_disability",
                    "other_gender",
                    "male",
                    "other_race_or_ethnicity",
                    "muslim",
                    "white",
                    "psychiatric_or_mental_illness",
                    "likes",
                    "transgender",
                    "disagree",
                    "latino",
                    "funny",
                    "atheist",
                    "sad",
                    "intellectual_or_learning_disability",
                    "wow",
                    "black",
                    "jewish",
                    "bisexual",
                    "asian",
                ],
                "disbert": [
                    "comment_text",
                    "white",
                    "other_religion",
                    "muslim",
                    "other_race_or_ethnicity",
                    "physical_disability",
                    "transgender",
                    "other_sexual_orientation",
                    "buddhist",
                    "other_disability",
                    "other_gender",
                    "male",
                    "funny",
                    "christian",
                    "disagree",
                    "latino",
                    "psychiatric_or_mental_illness",
                    "likes",
                    "jewish",
                    "intellectual_or_learning_disability",
                    "homosexual_gay_or_lesbian",
                    "wow",
                    "hindu",
                    "heterosexual",
                    "female",
                    "sad",
                    "atheist",
                    "asian",
                    "black",
                    "bisexual",
                ],
                "drob": [
                    "comment_text",
                    "hindu",
                    "homosexual_gay_or_lesbian",
                    "disagree",
                    "physical_disability",
                    "likes",
                    "other_sexual_orientation",
                    "psychiatric_or_mental_illness",
                    "christian",
                    "transgender",
                    "female",
                    "heterosexual",
                    "white",
                    "other_gender",
                    "black",
                    "other_religion",
                    "bisexual",
                    "funny",
                    "atheist",
                    "asian",
                    "buddhist",
                    "latino",
                    "male",
                    "sad",
                    "wow",
                    "other_race_or_ethnicity",
                    "intellectual_or_learning_disability",
                    "other_disability",
                    "jewish",
                    "muslim",
                ],
                "deberta": [
                    "comment_text",
                    "likes",
                    "disagree",
                    "white",
                    "wow",
                    "sad",
                    "funny",
                    "transgender",
                    "psychiatric_or_mental_illness",
                    "other_sexual_orientation",
                    "other_religion",
                    "physical_disability",
                    "asian",
                    "atheist",
                    "jewish",
                    "heterosexual",
                    "black",
                    "male",
                    "homosexual_gay_or_lesbian",
                    "bisexual",
                    "intellectual_or_learning_disability",
                    "female",
                    "muslim",
                    "latino",
                    "christian",
                    "hindu",
                    "buddhist",
                    "other_disability",
                    "other_gender",
                    "other_race_or_ethnicity",
                ],
            },
        )
    elif ds_type in ["wine", "wine_reviews"]:
        match model_type:
            case None:
                text_model_name = None
                ds_name = None
                warn(
                    f"No model type specified for {ds_type}. (This is fine during dataset creation)"
                )
            case "all_text" | "all_as_text":
                print(f"Using dataset {ds_type}, all as text version")
                text_model_name = "james-burton/wine_0"
                ds_name = "james-burton/wine_reviews_all_text"
            case _:
                print(f"Using dataset {ds_type}, ordinal version")
                text_model_name = "james-burton/wine_9"
                ds_name = "james-burton/wine_reviews_ordinal"
        return DatasetInfo(
            ds_name=ds_name,
            tab_cols=["points", "price"],
            categorical_cols=[],
            text_cols=["country", "description", "province"],
            label_col="variety",
            num_labels=30,
            prob_type="single_label_classification",
            wandb_proj_name="Wine",
            text_model_name=text_model_name,
            label_names=[
                "Bordeaux-style Red Blend",
                "Bordeaux-style White Blend",
                "Cabernet Franc",
                "Cabernet Sauvignon",
                "Champagne Blend",
                "Chardonnay",
                "Gamay",
                "Gewürztraminer",
                "Grüner Veltliner",
                "Malbec",
                "Merlot",
                "Nebbiolo",
                "Pinot Grigio",
                "Pinot Gris",
                "Pinot Noir",
                "Portuguese Red",
                "Portuguese White",
                "Red Blend",
                "Rhône-style Red Blend",
                "Riesling",
                "Rosé",
                "Sangiovese",
                "Sauvignon Blanc",
                "Shiraz",
                "Sparkling Blend",
                "Syrah",
                "Tempranillo",
                "Viognier",
                "White Blend",
                "Zinfandel",
            ],
            tnt_reorder_cols={
                "bert": ["description", "province", "country", "price", "points"],
                "disbert": ["description", "province", "country", "price", "points"],
                "drob": ["description", "province", "country", "price", "points"],
                "deberta": ["description", "province", "country", "price", "points"],
            },
            base_reorder_cols={
                "bert": ["description", "province", "country", "price", "points"],
                "disbert": ["description", "province", "country", "price", "points"],
                "drob": ["description", "province", "country", "price", "points"],
                "deberta": ["description", "province", "price", "country", "points"],
            },
        )
    elif ds_type in ["salary", "data_scientist_salary"]:
        match model_type:
            case None:
                text_model_name = None
                ds_name = None
                warn(
                    f"No model type specified for {ds_type}. (This is fine during dataset creation)"
                )
            case "all_text" | "all_as_text":
                print(f"Using dataset {ds_type}, all as text version")
                text_model_name = "james-burton/salary_0"
                ds_name = "james-burton/data_scientist_salary_all_text"
            case _:
                print(f"Using dataset {ds_type}, ordinal version")
                text_model_name = "james-burton/salary_9"
                ds_name = "james-burton/data_scientist_salary_ordinal"
        return DatasetInfo(
            ds_name=ds_name,
            tab_cols=["job_type"],
            categorical_cols=["job_type"],
            text_cols=[
                "experience",
                "job_description",
                "job_desig",
                "key_skills",
                "location",
            ],
            label_col="salary",
            num_labels=6,
            prob_type="single_label_classification",
            wandb_proj_name="Salary",
            text_model_name=text_model_name,
            label_names=["6to10", "10to15", "0to3", "15to25", "3to6", "25to50"],
            base_reorder_cols={
                "bert": [
                    "experience",
                    "job_description",
                    "job_desig",
                    "key_skills",
                    "location",
                    "job_type",
                ],
                "disbert": [
                    "experience",
                    "job_description",
                    "key_skills",
                    "job_desig",
                    "location",
                    "job_type",
                ],
                "drob": [
                    "experience",
                    "job_desig",
                    "key_skills",
                    "job_description",
                    "job_type",
                    "location",
                ],
                "deberta": [
                    "experience",
                    "job_description",
                    "key_skills",
                    "job_desig",
                    "location",
                    "job_type",
                ],
            },
            tnt_reorder_cols={
                "bert": [
                    "experience",
                    "job_desig",
                    "key_skills",
                    "job_description",
                    "location",
                    "job_type",
                ],
                "disbert": [
                    "experience",
                    "job_description",
                    "location",
                    "job_desig",
                    "key_skills",
                    "job_type",
                ],
                "drob": [
                    "experience",
                    "job_desig",
                    "key_skills",
                    "job_description",
                    "job_type",
                    "location",
                ],
                "deberta": [
                    "experience",
                    "key_skills",
                    "job_desig",
                    "job_description",
                    "location",
                    "job_type",
                ],
            },
        )
    elif ds_type in ["airbnb", "melbourne_airbnb"]:
        match model_type:
            case None:
                text_model_name = None
                ds_name = None
                warn(
                    f"No model type specified for {ds_type}. (This is fine during dataset creation)"
                )
            case "all_text" | "all_as_text":
                print(f"Using dataset {ds_type}, all as text version")
                text_model_name = "james-burton/airbnb_0"
                ds_name = "james-burton/melbourne_airbnb_all_text"
            case _:
                print(f"Using dataset {ds_type}, ordinal version")
                text_model_name = "james-burton/airbnb_9"
                ds_name = "james-burton/melbourne_airbnb_ordinal"
        return DatasetInfo(
            ds_name=ds_name,
            tab_cols=[
                "accommodates",
                "availability_30",
                "availability_365",
                "availability_60",
                "availability_90",
                "bathrooms",
                "bed_type",
                "bedrooms",
                "beds",
                # "calculated_host_listings_count",
                "cancellation_policy",
                "city",
                "cleaning_fee",
                "extra_people",
                "guests_included",
                # "host_has_profile_pic",
                "host_identity_verified",
                "host_is_superhost",
                "host_response_time",
                # "host_verifications_email",
                # "host_verifications_facebook",
                # "host_verifications_google",
                # "host_verifications_government_id",
                # "host_verifications_identity_manual",
                # "host_verifications_jumio",
                # "host_verifications_kba",
                # "host_verifications_manual_offline",
                # "host_verifications_manual_online",
                # "host_verifications_offline_government_id",
                # "host_verifications_phone",
                # "host_verifications_reviews",
                # "host_verifications_selfie",
                # "host_verifications_sent_id",
                # "host_verifications_sesame",
                # "host_verifications_sesame_offline",
                # "host_verifications_weibo",
                # "host_verifications_work_email",
                # "host_verifications_zhima_selfie",
                "instant_bookable",
                "is_location_exact",
                "latitude",
                "license",
                "longitude",
                "maximum_nights",
                "minimum_nights",
                "number_of_reviews",
                # "require_guest_phone_verification",
                # "require_guest_profile_picture",
                "review_scores_accuracy",
                "review_scores_checkin",
                "review_scores_cleanliness",
                "review_scores_communication",
                "review_scores_location",
                "review_scores_rating",
                "review_scores_value",
                "reviews_per_month",
                "room_type",
                "security_deposit",
            ],
            categorical_cols=[
                "bed_type",
                "cancellation_policy",
                "city",
                "host_identity_verified",
                "host_is_superhost",
                "host_response_time",
                # "host_verifications_email",
                # "host_verifications_facebook",
                # "host_verifications_google",
                # "host_verifications_government_id",
                # "host_verifications_identity_manual",
                # "host_verifications_jumio",
                # "host_verifications_kba",
                # "host_verifications_manual_offline",
                # "host_verifications_manual_online",
                # "host_verifications_offline_government_id",
                # "host_verifications_phone",
                # "host_verifications_reviews",
                # "host_verifications_selfie",
                # "host_verifications_sent_id",
                # "host_verifications_sesame",
                # "host_verifications_sesame_offline",
                # "host_verifications_weibo",
                # "host_verifications_work_email",
                # "host_verifications_zhima_selfie",
                "instant_bookable",
                "is_location_exact",
                "license",
                # "require_guest_phone_verification",
                # "require_guest_profile_picture",
                "room_type",
            ],
            text_cols=[
                # "access",
                "amenities",
                # "calendar_updated",
                # "description",
                # "first_review",
                # "host_about",
                # "host_location",
                # "host_neighborhood",
                # "host_response_rate",
                # "host_response_time",
                "host_since",
                # "host_verifications",
                # "house_rules",
                # "interaction",
                # "last_review",
                # "name",
                "neighborhood",
                # "neighborhood_overview",
                "property_type",
                # "smart_location",
                # "space",
                # "state",
                # "street",
                # "suburb",
                "summary",
                # "transit",
                # "zipcode",
            ],
            label_col="price_label",
            num_labels=10,
            prob_type="single_label_classification",
            wandb_proj_name="Airbnb",
            text_model_name=text_model_name,
            label_names=[4, 3, 1, 9, 2, 7, 0, 6, 5, 8],
            base_reorder_cols={
                "bert": [
                    "summary",
                    "amenities",
                    "room_type",
                    "security_deposit",
                    "bedrooms",
                    "city",
                    "bed_type",
                    "reviews_per_month",
                    "accommodates",
                    "cancellation_policy",
                    "bathrooms",
                    "cleaning_fee",
                    "beds",
                    "host_since",
                    "neighborhood",
                    "latitude",
                    "longitude",
                    "host_response_time",
                    "property_type",
                    "availability_60",
                    "availability_90",
                    "review_scores_rating",
                    "review_scores_location",
                    "guests_included",
                    "host_identity_verified",
                    "review_scores_communication",
                    "review_scores_value",
                    "host_is_superhost",
                    "review_scores_checkin",
                    "review_scores_cleanliness",
                    "review_scores_accuracy",
                    "availability_365",
                    "extra_people",
                    "availability_30",
                    "instant_bookable",
                    "maximum_nights",
                    "is_location_exact",
                    "number_of_reviews",
                    "minimum_nights",
                    "license",
                ],
                "disbert": [
                    "room_type",
                    "amenities",
                    "summary",
                    "bedrooms",
                    "beds",
                    "accommodates",
                    "reviews_per_month",
                    "city",
                    "bed_type",
                    "bathrooms",
                    "security_deposit",
                    "cancellation_policy",
                    "cleaning_fee",
                    "latitude",
                    "longitude",
                    "host_since",
                    "neighborhood",
                    "availability_90",
                    "availability_60",
                    "availability_365",
                    "review_scores_value",
                    "property_type",
                    "availability_30",
                    "review_scores_rating",
                    "review_scores_communication",
                    "review_scores_location",
                    "host_response_time",
                    "review_scores_cleanliness",
                    "review_scores_checkin",
                    "review_scores_accuracy",
                    "maximum_nights",
                    "extra_people",
                    "license",
                    "guests_included",
                    "host_identity_verified",
                    "minimum_nights",
                    "number_of_reviews",
                    "host_is_superhost",
                    "instant_bookable",
                    "is_location_exact",
                ],
                "drob": [
                    "room_type",
                    "amenities",
                    "summary",
                    "bedrooms",
                    "latitude",
                    "bathrooms",
                    "beds",
                    "bed_type",
                    "reviews_per_month",
                    "availability_90",
                    "accommodates",
                    "host_since",
                    "is_location_exact",
                    "cancellation_policy",
                    "longitude",
                    "city",
                    "security_deposit",
                    "cleaning_fee",
                    "host_response_time",
                    "review_scores_accuracy",
                    "review_scores_value",
                    "neighborhood",
                    "review_scores_rating",
                    "review_scores_checkin",
                    "review_scores_cleanliness",
                    "review_scores_communication",
                    "property_type",
                    "availability_365",
                    "guests_included",
                    "availability_30",
                    "review_scores_location",
                    "availability_60",
                    "maximum_nights",
                    "number_of_reviews",
                    "minimum_nights",
                    "instant_bookable",
                    "extra_people",
                    "license",
                    "host_is_superhost",
                    "host_identity_verified",
                ],
                "deberta": [
                    "room_type",
                    "summary",
                    "amenities",
                    "bathrooms",
                    "reviews_per_month",
                    "beds",
                    "bedrooms",
                    "bed_type",
                    "host_since",
                    "accommodates",
                    "security_deposit",
                    "latitude",
                    "availability_365",
                    "longitude",
                    "host_response_time",
                    "property_type",
                    "cancellation_policy",
                    "cleaning_fee",
                    "review_scores_value",
                    "neighborhood",
                    "city",
                    "availability_60",
                    "availability_90",
                    "availability_30",
                    "review_scores_checkin",
                    "review_scores_cleanliness",
                    "review_scores_communication",
                    "review_scores_location",
                    "review_scores_accuracy",
                    "review_scores_rating",
                    "license",
                    "instant_bookable",
                    "host_is_superhost",
                    "maximum_nights",
                    "guests_included",
                    "is_location_exact",
                    "host_identity_verified",
                    "minimum_nights",
                    "number_of_reviews",
                    "extra_people",
                ],
            },
            tnt_reorder_cols={
                "bert": [
                    "room_type",
                    "summary",
                    "bedrooms",
                    "amenities",
                    "city",
                    "accommodates",
                    "beds",
                    "property_type",
                    "neighborhood",
                    "bathrooms",
                    "latitude",
                    "security_deposit",
                    "cleaning_fee",
                    "host_since",
                    "availability_90",
                    "availability_60",
                    "availability_365",
                    "availability_30",
                    "review_scores_location",
                    "review_scores_communication",
                    "review_scores_checkin",
                    "review_scores_cleanliness",
                    "longitude",
                    "review_scores_value",
                    "review_scores_rating",
                    "review_scores_accuracy",
                    "cancellation_policy",
                    "extra_people",
                    "host_identity_verified",
                    "license",
                    "guests_included",
                    "reviews_per_month",
                    "number_of_reviews",
                    "host_is_superhost",
                    "host_response_time",
                    "maximum_nights",
                    "minimum_nights",
                    "instant_bookable",
                    "bed_type",
                    "is_location_exact",
                ],
                "disbert": [
                    "room_type",
                    "summary",
                    "amenities",
                    "bedrooms",
                    "beds",
                    "accommodates",
                    "city",
                    "bathrooms",
                    "neighborhood",
                    "cleaning_fee",
                    "latitude",
                    "security_deposit",
                    "property_type",
                    "availability_365",
                    "availability_90",
                    "availability_60",
                    "availability_30",
                    "host_since",
                    "cancellation_policy",
                    "longitude",
                    "review_scores_location",
                    "review_scores_value",
                    "review_scores_rating",
                    "review_scores_accuracy",
                    "review_scores_cleanliness",
                    "review_scores_checkin",
                    "review_scores_communication",
                    "license",
                    "host_identity_verified",
                    "host_response_time",
                    "extra_people",
                    "guests_included",
                    "reviews_per_month",
                    "number_of_reviews",
                    "host_is_superhost",
                    "maximum_nights",
                    "is_location_exact",
                    "minimum_nights",
                    "instant_bookable",
                    "bed_type",
                ],
                "drob": [
                    "room_type",
                    "amenities",
                    "summary",
                    "bedrooms",
                    "beds",
                    "accommodates",
                    "bathrooms",
                    "city",
                    "neighborhood",
                    "availability_90",
                    "availability_60",
                    "availability_30",
                    "availability_365",
                    "longitude",
                    "extra_people",
                    "cleaning_fee",
                    "guests_included",
                    "host_response_time",
                    "security_deposit",
                    "property_type",
                    "host_since",
                    "reviews_per_month",
                    "number_of_reviews",
                    "cancellation_policy",
                    "host_is_superhost",
                    "latitude",
                    "license",
                    "review_scores_location",
                    "review_scores_cleanliness",
                    "review_scores_value",
                    "review_scores_rating",
                    "review_scores_accuracy",
                    "review_scores_communication",
                    "review_scores_checkin",
                    "host_identity_verified",
                    "maximum_nights",
                    "minimum_nights",
                    "is_location_exact",
                    "instant_bookable",
                    "bed_type",
                ],
                "deberta": [
                    "room_type",
                    "summary",
                    "amenities",
                    "bedrooms",
                    "accommodates",
                    "beds",
                    "host_since",
                    "property_type",
                    "bathrooms",
                    "neighborhood",
                    "city",
                    "security_deposit",
                    "cleaning_fee",
                    "availability_365",
                    "reviews_per_month",
                    "number_of_reviews",
                    "availability_90",
                    "availability_60",
                    "availability_30",
                    "host_is_superhost",
                    "longitude",
                    "latitude",
                    "host_response_time",
                    "license",
                    "extra_people",
                    "review_scores_location",
                    "host_identity_verified",
                    "guests_included",
                    "review_scores_value",
                    "review_scores_rating",
                    "review_scores_accuracy",
                    "review_scores_cleanliness",
                    "review_scores_communication",
                    "review_scores_checkin",
                    "cancellation_policy",
                    "is_location_exact",
                    "maximum_nights",
                    "minimum_nights",
                    "instant_bookable",
                    "bed_type",
                ],
            },
        )
    elif ds_type in ["channel", "news_channel"]:
        match model_type:
            case None:
                text_model_name = None
                ds_name = None
                warn(
                    f"No model type specified for {ds_type}. (This is fine during dataset creation)"
                )
            case "all_text" | "all_as_text":
                print(f"Using dataset {ds_type}, all as text version")
                text_model_name = "james-burton/channel_0"
                ds_name = "james-burton/news_channel_all_text"
            case _:
                print(f"Using dataset {ds_type}, ordinal version")
                text_model_name = "james-burton/channel_9"
                ds_name = "james-burton/news_channel_ordinal"
        return DatasetInfo(
            ds_name=ds_name,
            tab_cols=[
                " n_tokens_content",
                " n_unique_tokens",
                " n_non_stop_words",
                " n_non_stop_unique_tokens",
                " num_hrefs",
                " num_self_hrefs",
                " num_imgs",
                " num_videos",
                " average_token_length",
                " num_keywords",
                " global_subjectivity",
                " global_sentiment_polarity",
                " global_rate_positive_words",
                " global_rate_negative_words",
                " rate_positive_words",
                " rate_negative_words",
            ],
            categorical_cols=[],
            text_cols=[
                "article_title",
            ],
            label_col="channel",
            num_labels=6,
            prob_type="single_label_classification",
            wandb_proj_name="News Channel",
            text_model_name=text_model_name,
            label_names=[
                " data_channel_is_tech",
                " data_channel_is_entertainment",
                " data_channel_is_lifestyle",
                " data_channel_is_bus",
                " data_channel_is_world",
                " data_channel_is_socmed",
            ],
            base_reorder_cols={
                "bert": [
                    " n_non_stop_words",
                    " n_unique_tokens",
                    " n_non_stop_unique_tokens",
                    " global_rate_positive_words",
                    " global_rate_negative_words",
                    " global_sentiment_polarity",
                    " global_subjectivity",
                    " rate_positive_words",
                    " average_token_length",
                    " rate_negative_words",
                    " n_tokens_content",
                    "article_title",
                    " num_hrefs",
                    " num_self_hrefs",
                    " num_videos",
                    " num_imgs",
                    " num_keywords",
                ],
                "disbert": [
                    "article_title",
                    " global_rate_positive_words",
                    " n_non_stop_unique_tokens",
                    " global_rate_negative_words",
                    " rate_negative_words",
                    " rate_positive_words",
                    " n_non_stop_words",
                    " num_hrefs",
                    " global_sentiment_polarity",
                    " n_unique_tokens",
                    " num_self_hrefs",
                    " global_subjectivity",
                    " average_token_length",
                    " num_imgs",
                    " num_videos",
                    " n_tokens_content",
                    " num_keywords",
                ],
                "drob": [
                    "article_title",
                    " global_sentiment_polarity",
                    " n_non_stop_words",
                    " rate_negative_words",
                    " global_rate_positive_words",
                    " n_unique_tokens",
                    " rate_positive_words",
                    " average_token_length",
                    " n_non_stop_unique_tokens",
                    " global_subjectivity",
                    " global_rate_negative_words",
                    " n_tokens_content",
                    " num_videos",
                    " num_imgs",
                    " num_keywords",
                    " num_hrefs",
                    " num_self_hrefs",
                ],
                "deberta": [
                    "article_title",
                    " global_sentiment_polarity",
                    " rate_negative_words",
                    " n_unique_tokens",
                    " global_rate_negative_words",
                    " global_rate_positive_words",
                    " average_token_length",
                    " n_non_stop_unique_tokens",
                    " global_subjectivity",
                    " rate_positive_words",
                    " n_non_stop_words",
                    " n_tokens_content",
                    " num_hrefs",
                    " num_self_hrefs",
                    " num_imgs",
                    " num_keywords",
                    " num_videos",
                ],
            },
            tnt_reorder_cols={
                "bert": [
                    "article_title",
                    " global_rate_negative_words",
                    " global_subjectivity",
                    " rate_negative_words",
                    " global_rate_positive_words",
                    " n_unique_tokens",
                    " rate_positive_words",
                    " n_non_stop_unique_tokens",
                    " average_token_length",
                    " global_sentiment_polarity",
                    " num_keywords",
                    " n_non_stop_words",
                    " num_videos",
                    " num_imgs",
                    " n_tokens_content",
                    " num_self_hrefs",
                    " num_hrefs",
                ],
                "disbert": [
                    "article_title",
                    " rate_positive_words",
                    " rate_negative_words",
                    " global_sentiment_polarity",
                    " global_rate_negative_words",
                    " global_rate_positive_words",
                    " global_subjectivity",
                    " n_unique_tokens",
                    " average_token_length",
                    " n_non_stop_unique_tokens",
                    " n_non_stop_words",
                    " n_tokens_content",
                    " num_hrefs",
                    " num_self_hrefs",
                    " num_imgs",
                    " num_videos",
                    " num_keywords",
                ],
                "drob": [
                    "article_title",
                    " global_sentiment_polarity",
                    " rate_negative_words",
                    " rate_positive_words",
                    " global_rate_positive_words",
                    " average_token_length",
                    " n_tokens_content",
                    " global_subjectivity",
                    " global_rate_negative_words",
                    " n_non_stop_unique_tokens",
                    " n_unique_tokens",
                    " n_non_stop_words",
                    " num_hrefs",
                    " num_self_hrefs",
                    " num_imgs",
                    " num_keywords",
                    " num_videos",
                ],
                "deberta": [
                    "article_title",
                    " global_sentiment_polarity",
                    " rate_negative_words",
                    " rate_positive_words",
                    " global_rate_negative_words",
                    " global_rate_positive_words",
                    " global_subjectivity",
                    " average_token_length",
                    " n_non_stop_unique_tokens",
                    " n_unique_tokens",
                    " n_non_stop_words",
                    " n_tokens_content",
                    " num_videos",
                    " num_self_hrefs",
                    " num_imgs",
                    " num_hrefs",
                    " num_keywords",
                ],
            },
        )
