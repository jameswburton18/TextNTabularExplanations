# %%
import shap
import pickle
from transformers import AutoTokenizer
import numpy as np
import transformers
import shap
from src.plot_text import text
from src.utils import format_fts_for_plotting, text_ft_index_ends
from src.utils import legacy_get_dataset_info
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from src.utils import token_segments


# %% [markdown]
# # Text style explanations


# %%
def load_shap_vals(ds_name):
    with open(f"models/shap_vals/{ds_name}/shap_vals_ensemble_25.pkl", "rb") as f:
        shap_25 = pickle.load(f)
    with open(f"models/shap_vals/{ds_name}/shap_vals_ensemble_50.pkl", "rb") as f:
        shap_50 = pickle.load(f)
    with open(f"models/shap_vals/{ds_name}/shap_vals_ensemble_75.pkl", "rb") as f:
        shap_75 = pickle.load(f)
    with open(f"models/shap_vals/{ds_name}/shap_vals_stack.pkl", "rb") as f:
        shap_stack = pickle.load(f)
    with open(f"models/shap_vals/{ds_name}/shap_vals_all_text.pkl", "rb") as f:
        shap_all_text = pickle.load(f)
    with open(f"models/shap_vals/{ds_name}/shap_vals_all_text_baseline.pkl", "rb") as f:
        shap_all_text_baseline = pickle.load(f)
    return (
        [shap_25, shap_50, shap_75, shap_stack, shap_all_text, shap_all_text_baseline],
        [
            "Ensemble 25",
            "Ensemble 50",
            "Ensemble 75",
            "Stack Ensemble",
            "All as Text",
            "All as Text Baseline",
        ],
    )


def get_line_breaks(ds_name):
    di = legacy_get_dataset_info(ds_name, "all_text")
    ds = load_dataset(di.ds_name, split="test")
    # Define how to convert all columns to a single string
    cols_to_str_fn = lambda array: " [SEP] ".join(
        [f"{col}: {val}" for col, val in zip(di.tab_cols + di.text_cols, array)]
    )
    examples_as_strings = np.apply_along_axis(cols_to_str_fn, 1, ds.to_pandas())
    list(map(cols_to_str_fn, ds.to_pandas()[di.text_cols].values))
    print("Getting linebreaks")


def display_shap_vals(ds_name, shap_groups, names, idx):
    di = legacy_get_dataset_info(ds_name)
    # names = ["Ensemble 25", "Ensemble 50",
    #         "Ensemble 75", "Stack Ensemble", "All as Text"]
    # shap_groups = [shap_25, shap_50, shap_75, shap_stack, shap_all_text]

    for shap_vals, name in zip(shap_groups[:-1], names[:-1]):
        print(
            f"""
            #################
            {name}
            #################
            """
        )
        # To format the text features, we find when the text features end and therefore
        # where to insert linebreaks
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        text_idxs = text_ft_index_ends(
            text_fts=shap_vals.data[idx][len(di.tab_cols) :], tokenizer=tokenizer
        )
        linebreak_before_idxs = [len(di.tab_cols)] + [
            x + len(di.tab_cols) + 1 for x in text_idxs
        ]

        formatted_data = np.array(
            format_fts_for_plotting(
                shap_vals[idx].feature_names, shap_vals[idx].data[: len(di.tab_cols)]
            )
        )
        text(
            shap.Explanation(
                values=shap_vals[idx].values,
                base_values=shap_vals[idx].base_values,
                data=formatted_data,
                clustering=shap_vals[idx].clustering,
                output_names=di.label_names,
                hierarchical_values=shap_vals[idx].hierarchical_values,
            ),
            # grouping_threshold=20,
            linebreak_before_idxs=linebreak_before_idxs,  # linebreak_after_idx,
            text_cols=di.text_cols,
            grouping_threshold=10,
        )
    print(
        f"""
            #################
            All as Text Baseline
            #################
            """
    )
    shap_vals = shap_groups[-1]
    shap.plots.text(
        shap.Explanation(
            values=shap_vals[idx].values,
            base_values=shap_vals[idx].base_values,
            data=shap_vals[idx].data,
            clustering=shap_vals[idx].clustering,
            output_names=di.label_names,
            hierarchical_values=shap_vals[idx].hierarchical_values,
        ),
        grouping_threshold=1,
    )


def cap_data_length(data, max_len):
    new_data = []
    for x in data:
        if len(str(x)) > max_len:
            new_data.append(str(x)[:max_len] + "...")
        else:
            new_data.append(x)
    return new_data


def waterfall_plot(shap_groups, names, idx, label, di, tokenizer):
    for shap_vals, name in zip(shap_groups[:-1], names[:-1]):
        print(
            f"""
            #################
            {name}
            #################
            """
        )
        sv = shap_vals[idx, :, label]
        text_ft_ends = text_ft_index_ends(sv.data[len(di.tab_cols) :], tokenizer)
        text_ft_ends = [len(di.tab_cols)] + [
            x + len(di.tab_cols) + 1 for x in text_ft_ends
        ]
        shap.waterfall_plot(
            shap.Explanation(
                values=np.append(
                    sv.values[: len(di.tab_cols)],
                    [
                        np.sum(sv.values[text_ft_ends[i] : text_ft_ends[i + 1]])
                        for i in range(len(text_ft_ends) - 1)
                    ]
                    + [np.sum(sv.values[text_ft_ends[-1] :])],
                ),
                base_values=sv.base_values,
                data=cap_data_length(sv.data, 100),
                feature_names=di.tab_cols + di.text_cols,
                # feature_names=shap_vals[idx, :len(di.tab_cols), label].feature_names
                # + ["Sum of text fts"],
            )
        )
    print(
        f"""
            #################
            All as Text Baseline
            #################
            """
    )
    shap_vals = shap_groups[-1]
    sv = shap_vals[idx, :, label]
    text_ft_ends = [1] + list(np.where(sv.data == "| ")[0]) + [len(sv.data) + 1]

    data = np.array(
        [
            "".join(sv.data[text_ft_ends[i] : text_ft_ends[i + 1]])
            for i in range(len(text_ft_ends) - 1)
        ]
    )
    capped_data = cap_data_length(data, 100)
    shap.waterfall_plot(
        shap.Explanation(
            values=np.array(
                [
                    np.sum(sv.values[text_ft_ends[i] : text_ft_ends[i + 1]])
                    for i in range(len(text_ft_ends) - 1)
                ]
            ),
            base_values=sv.base_values,
            data=capped_data,
            feature_names=di.tab_cols + di.text_cols,
        )
    )


def gen_summary_shap_vals(ds_name):
    names = [
        "ensemble_25",
        "ensemble_50",
        "ensemble_75",
        "stack",
        "all_text",
        "all_text_baseline",
    ]
    di = legacy_get_dataset_info(ds_name)
    shap_groups, _ = load_shap_vals(ds_name)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    for shap_vals, name in zip(shap_groups[:-1], names[:-1]):
        print(
            f"""
            #################
            {name}
            #################
            """
        )
        filepath = f"models/shap_vals/{ds_name}/shap_vals_{name}_summed.pkl"
        grouped_shap_vals = []
        for label in range(len(di.label_names)):
            shap_for_label = []
            for idx in tqdm(range(100)):
                sv = shap_vals[idx, :, label]
                text_ft_ends = text_ft_index_ends(
                    sv.data[len(di.tab_cols) :], tokenizer
                )
                text_ft_ends = [len(di.tab_cols)] + [
                    x + len(di.tab_cols) + 1 for x in text_ft_ends
                ]
                val = np.append(
                    sv.values[: len(di.tab_cols)],
                    [
                        np.sum(sv.values[text_ft_ends[i] : text_ft_ends[i + 1]])
                        for i in range(len(text_ft_ends) - 1)
                    ]
                    + [np.sum(sv.values[text_ft_ends[-1] :])],
                )

                shap_for_label.append(val)
            grouped_shap_vals.append(np.vstack(shap_for_label))
        print(f"Saving to {filepath}")
        with open(filepath, "wb") as f:
            pickle.dump(np.array(grouped_shap_vals), f)

    print(
        f"""
        #################
        All as Text Baseline
        #################
        """
    )
    shap_vals = shap_groups[-1]
    filepath = f"models/shap_vals/{ds_name}/shap_vals_all_text_baseline_summed.pkl"
    grouped_shap_vals = []
    for label in range(len(di.label_names)):
        shap_for_label = []
        for idx in tqdm(range(100)):
            sv = shap_vals[idx, :, label]
            text_ft_ends = [1] + list(np.where(sv.data == "| ")[0]) + [len(sv.data) + 1]
            # Need this if there are | in the text that aren't col separators
            if len(text_ft_ends) != len(di.text_cols + di.tab_cols) + 1:
                text_ft_ends = (
                    [1]
                    + [
                        i
                        for i in list(np.where(sv.data == "| ")[0])
                        if sv.data[i + 1]
                        in [
                            token_segments(col, tokenizer)[0][1]
                            for col in di.tab_cols + di.text_cols
                        ]
                    ]
                    + [len(sv.data) + 1]
                )
            val = np.array(
                [
                    np.sum(sv.values[text_ft_ends[i] : text_ft_ends[i + 1]])
                    for i in range(len(text_ft_ends) - 1)
                ]
            )
            shap_for_label.append(val)
        grouped_shap_vals.append(np.vstack(shap_for_label))
    print(f"Saving to {filepath}")
    with open(filepath, "wb") as f:
        pickle.dump(np.array(grouped_shap_vals), f)


def summary_plot(shap_groups, names, idx, label, di, tokenizer):
    for shap_vals, name in zip(shap_groups[:-1], names[:-1]):
        print(
            f"""
            #################
            {name}
            #################
            """
        )
        filepath = f"models/shap_vals/{ds}/shap_vals_{name}.pkl"
        grouped_shap_vals = []
        for label in range(len(di.label_names)):
            shap_for_label = []
            for idx in tqdm(range(100)):
                sv = shap_vals[idx, :, label]
                text_ft_ends = text_ft_index_ends(
                    sv.data[len(di.tab_cols) :], tokenizer
                )
                text_ft_ends = [len(di.tab_cols)] + [
                    x + len(di.tab_cols) + 1 for x in text_ft_ends
                ]
                val = np.append(
                    sv.values[: len(di.tab_cols)],
                    [
                        np.sum(sv.values[text_ft_ends[i] : text_ft_ends[i + 1]])
                        for i in range(len(text_ft_ends) - 1)
                    ]
                    + [np.sum(sv.values[text_ft_ends[-1] :])],
                )

                shap_for_label.append(val)
            grouped_shap_vals.append(np.vstack(shap_for_label))
        shap.summary_plot(
            grouped_shap_vals,
            features=[f"(Tab ft) {col}" for col in di.tab_cols]
            + [f"(Text ft) {col}" for col in di.text_cols],
            class_names=di.label_names,
        )


# %%
if __name__ == "__main__":
    for ds_name in [
        "prod_sent",
        "kick",
        "jigsaw",
        "wine",
        "fake",
        "imdb_genre",
    ]:  #
        # get_line_breaks(ds_name)
        gen_summary_shap_vals(ds_name)
