import numpy as np
from shap.utils import safe_isinstance
from shap.utils.transformers import (
    SENTENCEPIECE_TOKENIZERS,
    getattr_silent,
)
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


'''
# Deprecated
###############################################
def prepare_text_old(dataset, version, ds_type):
    if ds_type in ["imdb", "imdb_genre"]:

    elif "prod_sent" in ds_type:
        di = get_dataset_info("prod_sent")
        if version == "text_col_only":
            # dataset rename column
            dataset = dataset.rename_column(di.text_cols[0], "text")
            return dataset
        elif version == "all_as_text":
            cols = di.tab_cols + di.text_cols
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
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
                f"Unknown dataset type ({ds_type}) and version ({version}) combination"
            )
    elif "fake" in ds_type:
        di = get_dataset_info("fake")
        if version == "text_col_only":
            cols = di.text_cols
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
        elif version == "all_as_text":
            cols = di.tab_cols + di.text_cols

            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
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
                f"Unknown dataset type ({ds_type}) and version ({version}) combination"
            )
    elif "kick" in ds_type:
        di = get_dataset_info("kick")
        if version == "text_col_only":
            cols = di.text_cols
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
        elif version == "all_as_text":
            cols = di.tab_cols + di.text_cols

            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
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
                f"Unknown dataset type ({ds_type}) and version ({version}) combination"
            )
    elif "jigsaw" in ds_type:
        di = get_dataset_info("jigsaw")
        if version == "text_col_only":
            dataset = dataset.rename_column(di.text_cols[0], "text")
            return dataset
        elif version == "all_as_text":
            cols = di.tab_cols + di.text_cols

            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
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
                f"Unknown dataset type ({ds_type}) and version ({version}) combination"
            )
    elif "wine" in ds_type:
        di = get_dataset_info("wine")
        if version == "text_col_only":
            cols = di.text_cols
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
        elif version == "all_as_text":
            cols = di.tab_cols + di.text_cols

            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
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
                f"Unknown dataset type ({ds_type}) and version ({version}) combination"
            )
    elif "airbnb" in ds_type:
        di = get_dataset_info("airbnb")
        if version == "text_col_only":
            cols = di.text_cols
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
        elif version == "all_as_text":
            cols = di.tab_cols + di.text_cols
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
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
                f"Unknown dataset type ({ds_type}) and version ({version}) combination"
            )
    elif "channel" in ds_type:
        di = get_dataset_info("channel")
        if version == "text_col_only":
            cols = di.text_cols
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
        elif version == "all_as_text":
            cols = di.tab_cols + di.text_cols
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
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
                f"Unknown dataset type ({ds_type}) and version ({version}) combination"
            )
    elif "salary" in ds_type:
        di = get_dataset_info("salary")
        if version == "text_col_only":
            cols = di.text_cols
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
        elif version == "all_as_text":
            cols = di.tab_cols + di.text_cols
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
            return dataset
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
                f"Unknown dataset type ({ds_type}) and version ({version}) combination"
            )

    else:
        raise ValueError(
            f"Unknown dataset type ({ds_type}) and version ({version}) combination"
        )

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


"""
# Less used versions (IMDB)
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
"""
'''
