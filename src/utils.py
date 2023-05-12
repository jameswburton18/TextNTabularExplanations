import numpy as np
from shap.utils import safe_isinstance
from shap.utils.transformers import (
    SENTENCEPIECE_TOKENIZERS,
    getattr_silent,
)


MODEL_NAME_TO_DESC_DICT = {
    "imdb_genre_0": "all as text",
    "imdb_genre_1": "tab as text",
    "imdb_genre_2": "tabx5",
    "imdb_genre_3": "all as text, reorder1",
    "imdb_genre_4": "all as text, reorder2",
    "imdb_genre_5": "tabx5_nodesc",
    "imdb_genre_6": "tabx2_nodesc",
    "imdb_genre_7": "tabx2",
    "imdb_genre_8": "tab as text, exp-based reorder",
    "imdb_genre_9": "text col only",
    "imdb_genre_10": "all as text exp-based reorder",
}


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


def prepare_text(dataset, version, ds_type):
    """This is all for preparing the text part of the dataset
    Could be made more robust by referring to dataset_info.py instead"""
    if "imdb" in ds_type:
        if version == "all_as_text":
            cols = [
                "Year",
                "Runtime (Minutes)",
                "Rating",
                "Votes",
                "Revenue (Millions)",
                "Metascore",
                "Rank",
                "Description",
            ]
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
        elif version == "text_col_only":
            # dataset rename column
            dataset = dataset.rename_column("Description", "text")
            return dataset
        elif version == "all_as_text_base_reorder":
            cols = [
                "Description",
                "Metascore",
                "Runtime (Minutes)",
                "Revenue (Millions)",
                "Rank",
                "Rating",
                "Votes",
                "Year",
            ]
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
        elif version == "all_as_text_tnt_reorder":
            cols = [
                "Description",
                "Votes",
                "Rank",
                "Revenue (Millions)",
                "Runtime (Minutes)",
                "Year",
                "Metascore",
                "Rating",
            ]
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
        # Less used versions
        #########################################
        elif version == "tab_as_text":
            cols = [
                "Year",
                "Runtime (Minutes)",
                "Rating",
                "Votes",
                "Revenue (Millions)",
                "Metascore",
                "Rank",
            ]
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
        elif version == "tabx5":
            cols = [
                "Year",
                "Runtime (Minutes)",
                "Rating",
                "Votes",
                "Revenue (Millions)",
                "Metascore",
                "Rank",
            ]
            dataset = dataset.map(
                multiple_row_to_string, fn_kwargs={"cols": cols, "multiplier": 5}
            )
            return dataset
        elif version == "tabx5_nodesc":
            cols = [
                "Year",
                "Runtime (Minutes)",
                "Rating",
                "Votes",
                "Revenue (Millions)",
                "Metascore",
                "Rank",
            ]
            dataset = dataset.map(
                multiple_row_to_string,
                fn_kwargs={"cols": cols, "multiplier": 5, "nodesc": True},
            )
            return dataset
        elif version == "tabx2_nodesc":
            cols = [
                "Year",
                "Runtime (Minutes)",
                "Rating",
                "Votes",
                "Revenue (Millions)",
                "Metascore",
                "Rank",
            ]
            dataset = dataset.map(
                multiple_row_to_string,
                fn_kwargs={"cols": cols, "multiplier": 2, "nodesc": True},
            )
            return dataset
        elif version == "tabx2":
            cols = [
                "Year",
                "Runtime (Minutes)",
                "Rating",
                "Votes",
                "Revenue (Millions)",
                "Metascore",
                "Rank",
            ]
            dataset = dataset.map(
                multiple_row_to_string, fn_kwargs={"cols": cols, "multiplier": 2}
            )
            return dataset
        elif version == "reorder1":
            cols = [
                "Votes",
                "Revenue (Millions)",
                "Metascore",
                "Rank",
                "Description",
                "Year",
                "Runtime (Minutes)",
                "Rating",
            ]
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
        elif version == "reorder2":
            cols = [
                "Description",
                "Rank",
                "Metascore",
                "Revenue (Millions)",
                "Votes",
                "Rating",
                "Runtime (Minutes)",
                "Year",
            ]
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
        elif version == "reorder3":
            cols = [
                "Revenue (Millions)",
                "Metascore",
                "Rank",
                "Year",
                "Votes",
                "Runtime (Minutes)",
                "Rating",
            ]
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
        elif version == "all_as_text_exp_reorder":
            cols = [
                "Description",
                "Revenue (Millions)",
                "Votes",
                "Rank",
                "Metascore",
                "Year",
                "Runtime (Minutes)",
                "Rating",
            ]
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
        #########################################
        else:
            raise ValueError(
                f"Unknown dataset type ({ds_type}) and version ({version}) combination"
            )
    elif "prod_sent" in ds_type:
        if version == "text_col_only":
            # dataset rename column
            dataset = dataset.rename_column("Product_Description", "text")
            return dataset
        elif version == "all_as_text":
            cols = ["Product_Type", "Product_Description"]
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
        elif version == "all_as_text_base_reorder":
            raise Exception("Same order as original so not needed")
        elif version == "all_as_text_tnt_reorder":
            raise Exception("Same order as original so not needed")
        else:
            raise ValueError(
                f"Unknown dataset type ({ds_type}) and version ({version}) combination"
            )
    elif "fake" in ds_type:
        if version == "text_col_only":
            cols = ["title", "description", "salary_range"]
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
        elif version == "all_as_text":
            cols = [
                "title",
                "required_experience",
                "required_education",
                "title",
                "salary_range",
                "description",
            ]

            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
        elif version == "all_as_text_base_reorder":
            raise Exception(
                "Different order to original but same as text and tabular order so not needed"
            )
        elif version == "all_as_text_tnt_reorder":
            cols = [
                "description",
                "title",
                "required_education",
                "required_experience",
                "salary_range",
            ]
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
        else:
            raise ValueError(
                f"Unknown dataset type ({ds_type}) and version ({version}) combination"
            )
    elif "kick" in ds_type:
        if version == "text_col_only":
            cols = ["name", "desc", "keywords"]
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
        elif version == "all_as_text":
            cols = [
                "goal",
                "disable_communication",
                "country",
                "currency",
                "deadline",
                "created_at",
                "name",
                "desc",
                "keywords",
            ]

            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
        elif version == "all_as_text_base_reorder":
            cols = [
                "desc",
                "goal",
                "name",
                "created_at",
                "deadline",
                "keywords",
                "disable_communication",
                "currency",
                "country",
            ]
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
        elif version == "all_as_text_tnt_reorder":
            cols = [
                "desc",
                "goal",
                "name",
                "keywords",
                "created_at",
                "deadline",
                "currency",
                "country",
                "disable_communication",
            ]
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
        else:
            raise ValueError(
                f"Unknown dataset type ({ds_type}) and version ({version}) combination"
            )
    elif "jigsaw" in ds_type:
        if version == "text_col_only":
            dataset = dataset.rename_column("comment_text", "text")
            return dataset
        elif version == "all_as_text":
            cols = [
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
                "comment_text",
            ]

            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
        elif version == "all_as_text_base_reorder":
            cols = [
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
            ]
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
        elif version == "all_as_text_tnt_reorder":
            cols = [
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
            ]
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
        else:
            raise ValueError(
                f"Unknown dataset type ({ds_type}) and version ({version}) combination"
            )
    elif "wine" in ds_type:
        if version == "text_col_only":
            cols = ["country", "description", "province"]
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
        elif version == "all_as_text":
            cols = ["points", "price", "country", "description", "province"]

            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
        elif version == "all_as_text_base_reorder":
            raise Exception(
                "Different order to original but same as text and tabular order so not needed"
            )
        elif version == "all_as_text_tnt_reorder":
            cols = ["description", "province", "country", "price", "points"]
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
        else:
            raise ValueError(
                f"Unknown dataset type ({ds_type}) and version ({version}) combination"
            )

    else:
        raise ValueError(
            f"Unknown dataset type ({ds_type}) and version ({version}) combination"
        )


def format_text_pred(pred):
    # if pred["label"] == "LABEL_1":
    #     return np.array([1 - pred["score"], pred["score"]])
    # else:
    #     return np.array([pred["score"], 1 - pred["score"]])
    scores = [p["score"] for p in pred]
    order = [int(p["label"][6:]) for p in pred]
    return np.array([scores[i] for i in order])


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


# Deprecated
###############################################
def select_prepare_array_fn(model_name):
    if model_name == "imdb_genre_0":  # all as text
        cols = [
            "Year",
            "Runtime (Minutes)",
            "Rating",
            "Votes",
            "Revenue (Millions)",
            "Metascore",
            "Rank",
            "Description",
        ]

        def array_fn(array):
            return np.array(
                " | ".join([f"{col}: {val}" for col, val in zip(cols, array)]),
                dtype="<U512",
            )

    elif model_name == "imdb_genre_1":  # tab as text
        cols = [
            "Year",
            "Runtime (Minutes)",
            "Rating",
            "Votes",
            "Revenue (Millions)",
            "Metascore",
            "Rank",
        ]

        def array_fn(array):
            return np.array(
                " | ".join([f"{col}: {val}" for col, val in zip(cols, array)]),
                dtype="<U512",
            )

    elif model_name == "imdb_genre_6":  # tabx2_nodesc
        cols = [
            "Year",
            "Runtime (Minutes)",
            "Rating",
            "Votes",
            "Revenue (Millions)",
            "Metascore",
            "Rank",
        ]

        def array_fn(array):
            return np.array(
                " | ".join(
                    [f'{col}: {(str(val) + " ")*2}' for col, val in zip(cols, array)]
                ),
                dtype="<U512",
            )

    elif model_name == "imdb_genre_5":  # tabx5_nodesc
        cols = [
            "Year",
            "Runtime (Minutes)",
            "Rating",
            "Votes",
            "Revenue (Millions)",
            "Metascore",
            "Rank",
        ]

        def array_fn(array):
            return np.array(
                " | ".join(
                    [f'{col}: {(str(val) + " ")*5}' for col, val in zip(cols, array)]
                ),
                dtype="<U512",
            )

    elif model_name == "imdb_genre_7":  # tabx2
        cols = [
            "Year",
            "Runtime (Minutes)",
            "Rating",
            "Votes",
            "Revenue (Millions)",
            "Metascore",
            "Rank",
        ]

        def array_fn(array):
            return np.array(
                " | ".join(
                    [
                        f'{col}: {(str(val) + " ")*2}'
                        for col, val in zip(cols, array[:-1])
                    ]
                )
                + " | Description: "
                + array[-1],
                dtype="<U512",
            )

    elif model_name == "imdb_genre_2":  # tabx5
        cols = [
            "Year",
            "Runtime (Minutes)",
            "Rating",
            "Votes",
            "Revenue (Millions)",
            "Metascore",
            "Rank",
        ]

        def array_fn(array):
            return np.array(
                " | ".join(
                    [
                        f'{col}: {(str(val) + " ")*5}'
                        for col, val in zip(cols, array[:-1])
                    ]
                )
                + " | Description: "
                + array[-1],
                dtype="<U512",
            )

    elif model_name == "imdb_genre_3":  # reorder1
        cols = [
            "Votes",
            "Revenue (Millions)",
            "Metascore",
            "Rank",
            "Description",
            "Year",
            "Runtime (Minutes)",
            "Rating",
        ]

        def array_fn(array):
            return np.array(
                " | ".join([f"{col}: {val}" for col, val in zip(cols, array)]),
                dtype="<U512",
            )

    elif model_name == "imdb_genre_4":  # reorder2
        cols = [
            "Description",
            "Rank",
            "Metascore",
            "Revenue (Millions)",
            "Votes",
            "Rating",
            "Runtime (Minutes)",
            "Year",
        ]

        def array_fn(array):
            return np.array(
                " | ".join([f"{col}: {val}" for col, val in zip(cols, array)]),
                dtype="<U512",
            )

    elif model_name == "imdb_genre_8":  # reorder3/'tab as text, exp-based reorder'
        cols = [
            "Revenue (Millions)",
            "Metascore",
            "Rank",
            "Year",
            "Votes",
            "Runtime (Minutes)",
            "Rating",
        ]

        def array_fn(array):
            return np.array(
                " | ".join([f"{col}: {val}" for col, val in zip(cols, array)]),
                dtype="<U512",
            )

    elif model_name == "imdb_genre_10":  # all_as_text_exp_reorder
        cols = [
            "Description",
            "Revenue (Millions)",
            "Votes",
            "Rank",
            "Metascore",
            "Year",
            "Runtime (Minutes)",
            "Rating",
        ]

        def array_fn(array):
            return np.array(
                " | ".join([f"{col}: {val}" for col, val in zip(cols, array)]),
                dtype="<U512",
            )

    else:
        raise ValueError(f"select_prepare_array_fn not implemented for: {model_name}")
    return array_fn


def tokenize_lens(text_fts, tokenizer):
    return [len(tokenizer.tokenize(ft)) for ft in text_fts]


def array_to_string(
    array,
    cols=[
        "Year",
        "Runtime (Minutes)",
        "Rating",
        "Votes",
        "Revenue (Millions)",
        "Metascore",
        "Rank",
        "Description",
    ],
):
    return np.array(
        " | ".join([f"{col}: {val}" for col, val in zip(cols, array)]), dtype="<U512"
    )
