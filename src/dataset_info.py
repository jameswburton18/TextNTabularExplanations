from dataclasses import dataclass
from typing import List
from warnings import warn


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
    text_model_name: str = None


def get_dataset_info(ds_type, model_type=None):
    # if ds_type == "imdb_genre":imdb_genre_prediction
    #     return DatasetInfo(
    #         ds_name_all_text="james-burton/imdb_genre_prediction2",
    #         ds_name_ordinal="james-burton/imdb_genre_prediction2",
    #         tab_cols=[
    #             "Year",
    #             "Runtime (Minutes)",
    #             "Rating",
    #             "Votes",
    #             "Revenue (Millions)",
    #             "Metascore",
    #             "Rank",
    #         ],
    #         categorical_cols=[],
    #         text_cols=["Description"],
    #         label_col="Genre_is_Drama",
    #         num_labels=2,
    #         prob_type="single_label_classification",
    #         wandb_proj_name="IMDB Genre",
    #     )
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
        )
