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
    tnt_reorder_cols: List[str] = None
    base_reorder_cols: List[str] = None
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
            tnt_reorder_cols=[
                "Description",
                "Votes",
                "Rank",
                "Revenue (Millions)",
                "Runtime (Minutes)",
                "Year",
                "Metascore",
                "Rating",
            ],
            base_reorder_cols=[
                "Description",
                "Metascore",
                "Runtime (Minutes)",
                "Revenue (Millions)",
                "Rank",
                "Rating",
                "Votes",
                "Year",
            ],
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
            tnt_reorder_cols=[
                "description",
                "title",
                "required_education",
                "required_experience",
                "salary_range",
            ],
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
            tnt_reorder_cols=[
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
            base_reorder_cols=[
                "desc",
                "goal",
                "name",
                "created_at",
                "deadline",
                "keywords",
                "disable_communication",
                "currency",
                "country",
            ],
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
            tnt_reorder_cols=[
                "comment_text",
                "atheist",
                "buddhist",
                "hindu",
                "other_gender",
                "christian",
                "other_religion",
                "jewish",
                "muslim",
                "latino",
                "heterosexual",
                "black",
                "male",
                "female",
                "other_sexual_orientation",
                "other_race_or_ethnicity",
                "asian",
                "psychiatric_or_mental_illness",
                "white",
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
            base_reorder_cols=[
                "comment_text",
                "other_race_or_ethnicity",
                "other_sexual_orientation",
                "psychiatric_or_mental_illness",
                "other_religion",
                "intellectual_or_learning_disability",
                "physical_disability",
                "other_disability",
                "homosexual_gay_or_lesbian",
                "other_gender",
                "funny",
                "white",
                "buddhist",
                "muslim",
                "christian",
                "disagree",
                "transgender",
                "wow",
                "male",
                "sad",
                "latino",
                "jewish",
                "heterosexual",
                "female",
                "hindu",
                "likes",
                "atheist",
                "black",
                "bisexual",
                "asian",
            ],
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
            tnt_reorder_cols=["description", "province", "country", "price", "points"],
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
                text_model_name = "james-burton/melbourne_airbnb_0"
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
        )


"""
"""
