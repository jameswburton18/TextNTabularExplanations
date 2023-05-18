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

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ds_type",
    type=str,
    default="kick",
    help="Name of dataset to use",
)


def run_shap(
    model_type, ds_type, max_samples=100, test_set_size=100, tab_scale_factor=2
):
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
    test_df = test_df.sample(test_set_size, random_state=55)

    # Models
    tokenizer = AutoTokenizer.from_pretrained(
        # "distilbert-base-uncased"
        "distilroberta-base"
    )
    if model_type in [
        "all_text",
        "all_as_text_tnt_reorder",
        "all_as_text_base_reorder",
    ]:
        text_pipeline = pipeline(
            "text-classification",
            model=di.text_model_name[:-1] + "2" + di.text_model_name[-1],
            tokenizer=tokenizer,
            device="cuda:0",
            truncation=True,
            padding=True,
            top_k=None,
        )
        # Define how to convert all columns to a single string
        if model_type == "all_text":
            cols_to_str_fn = lambda array: " | ".join(
                [f"{col}: {val}" for col, val in zip(di.tab_cols + di.text_cols, array)]
            )
        else:
            # Reorder based on the new index order in di
            cols_to_str_fn = lambda array: " | ".join(
                [
                    f"{col}: {val}"
                    for _, col, val in sorted(
                        zip(di.new_idx_order, di.tab_cols + di.text_cols, array)
                    )
                ]
            )

        model = AllAsTextModel(
            text_pipeline=text_pipeline,
            cols_to_str_fn=cols_to_str_fn,
            # cols=di.tab_cols + di.text_cols
        )
    else:
        text_pipeline = pipeline(
            "text-classification",
            model=di.text_model_name[:-1] + "2" + di.text_model_name[-1],
            tokenizer=tokenizer,
            device="cuda:0",
            truncation=True,
            padding=True,
            top_k=None,
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
    x = test_df[di.tab_cols + di.text_cols].values
    # x = np.array(
    #     [
    #         [
    #             1800.0,
    #             0.0,
    #             9.0,
    #             8.0,
    #             1413695816,
    #             1406407374,
    #             "Romania: Timeless Beauty 2015 Calendar",
    #             "A calendar featuring the scenic and architectural beauty of Romania.",
    #             "romania-timeless-beauty-calendar",
    #         ]
    #     ],
    #     dtype=object,
    # )
    # x = np.array([[85, 25.0, "US", "tough and chewy", "Oregon"]], dtype=object)
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
        tab_cluster_scale_factor=tab_scale_factor,
    )

    explainer = shap.explainers.Partition(model=model.predict, masker=masker)
    shap_vals = explainer(x)

    pre = f"_sf{tab_scale_factor}" if tab_scale_factor != 2 else ""
    text_model_name = "_drob"

    output_dir = os.path.join(f"models/shap_vals{text_model_name}{pre}/", ds_type)
    print(f"Results will be saved @: {output_dir}")

    # Make output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, f"shap_vals_{model_type}.pkl"), "wb") as f:
        pickle.dump(shap_vals, f)

    return shap_vals


def run_all_text_baseline_shap(
    ds_type, max_samples=100, test_set_size=100, tab_scale_factor=2
):
    di = get_dataset_info(ds_type, "all_text")
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
    tokenizer = AutoTokenizer.from_pretrained(
        # "distilbert-base-uncased"
        "distilroberta-base"
    )
    text_pipeline = pipeline(
        "text-classification",
        model=di.text_model_name[:-1] + "2" + di.text_model_name[-1],
        tokenizer=tokenizer,
        device="cuda:0",
        truncation=True,
        padding=True,
        top_k=None,
    )
    # Define how to convert all columns to a single string
    cols_to_str_fn = lambda array: " | ".join(
        [f"{col}: {val}" for col, val in zip(di.tab_cols + di.text_cols, array)]
    )

    np.random.seed(1)
    x = list(map(cols_to_str_fn, test_df[di.tab_cols + di.text_cols].values))
    explainer = shap.Explainer(text_pipeline, tokenizer)
    shap_vals = explainer(x)

    # explainer = shap.explainers.Partition(model=model.predict, masker=tokenizer)
    # shap_vals = explainer(x)

    pre = f"_sf{tab_scale_factor}" if tab_scale_factor != 2 else ""
    text_model_name = "_drob"
    output_dir = os.path.join(f"models/shap_vals{text_model_name}{pre}/", ds_type)
    print(f"Results will be saved @: {output_dir}")

    # Make output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_type = "all_text_baseline"
    with open(os.path.join(output_dir, f"shap_vals_{model_type}.pkl"), "wb") as f:
        pickle.dump(shap_vals, f)

    return shap_vals


def load_shap_vals(ds_name, add_parent_dir=True, tab_scale_factor=2):
    pre = "../" if add_parent_dir else ""  # for running from notebooks
    tab_pre = f"_sf{tab_scale_factor}" if tab_scale_factor != 2 else ""
    text_model_name = "_drob"
    with open(
        f"{pre}models/shap_vals{text_model_name}{tab_pre}/{ds_name}/shap_vals_ensemble_25.pkl",
        "rb",
    ) as f:
        shap_25 = pickle.load(f)
    with open(
        f"{pre}models/shap_vals{text_model_name}{tab_pre}/{ds_name}/shap_vals_ensemble_50.pkl",
        "rb",
    ) as f:
        shap_50 = pickle.load(f)
    with open(
        f"{pre}models/shap_vals{text_model_name}{tab_pre}/{ds_name}/shap_vals_ensemble_75.pkl",
        "rb",
    ) as f:
        shap_75 = pickle.load(f)
    with open(
        f"{pre}models/shap_vals{text_model_name}{tab_pre}/{ds_name}/shap_vals_stack.pkl",
        "rb",
    ) as f:
        shap_stack = pickle.load(f)
    with open(
        f"{pre}models/shap_vals{text_model_name}{tab_pre}/{ds_name}/shap_vals_all_text.pkl",
        "rb",
    ) as f:
        shap_all_text = pickle.load(f)
    with open(
        f"{pre}models/shap_vals{text_model_name}{tab_pre}/{ds_name}/shap_vals_all_text_baseline.pkl",
        "rb",
    ) as f:
        shap_all_text_baseline = pickle.load(f)
    return (
        [shap_25, shap_50, shap_75, shap_stack, shap_all_text, shap_all_text_baseline],
        [
            "ensemble_25",
            "ensemble_50",
            "ensemble_75",
            "stack",
            "all_text",
            "all_text_baseline",
        ],
    )


def gen_summary_shap_vals(ds_name, add_parent_dir=False, tab_scale_factor=2):
    di = get_dataset_info(ds_name)
    shap_groups, names = load_shap_vals(
        ds_name, add_parent_dir, tab_scale_factor=tab_scale_factor
    )
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    pre = f"_sf{tab_scale_factor}" if tab_scale_factor != 2 else ""
    text_model_name = "_drob"

    for shap_vals, name in zip(shap_groups[:-1], names[:-1]):
        print(
            f"""
            #################
            {name}
            #################
            """
        )
        filepath = f"models/shap_vals{text_model_name}{pre}/{ds_name}/summed_shap_vals_{name}.pkl"
        grouped_shap_vals = []
        for label in range(len(di.label_names)):
            shap_for_label = []
            for idx in tqdm(range(len(shap_vals))):
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
    filepath = f"models/shap_vals{text_model_name}{pre}/{ds_name}/summed_shap_vals_all_text_baseline.pkl"
    col_name_filepath = f"models/shap_vals{text_model_name}{pre}/{ds_name}/col_names_shap_vals_all_text_baseline.pkl"
    colon_filepath = f"models/shap_vals{text_model_name}{pre}/{ds_name}/colon_shap_vals_all_text_baseline.pkl"
    grouped_shap_vals = []
    grouped_col_name_shap_vals = []
    grouped_colon_shap_vals = []
    for label in range(len(di.label_names)):
        shap_for_label = []
        shap_for_col_name = []
        shap_for_colon = []
        for idx in tqdm(range(len(shap_vals))):
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
            colon_idxs = np.where(sv.data == ": ")[0]
            col_idxs_after_ft = [
                colon_idxs[list(np.where(colon_idxs > te)[0])[0]]
                for te in text_ft_ends[:-1]
            ]
            ft_name_vals = np.array(
                [
                    np.sum(sv.values[text_ft_ends[i] : col_idxs_after_ft[i]])
                    for i in range(len(text_ft_ends) - 1)
                ]
            )
            colon_vals = np.array(sv.values[col_idxs_after_ft])
            shap_for_label.append(val)
            shap_for_col_name.append(ft_name_vals)
            shap_for_colon.append(colon_vals)
        grouped_shap_vals.append(np.vstack(shap_for_label))
        grouped_col_name_shap_vals.append(shap_for_col_name)
        grouped_colon_shap_vals.append(shap_for_colon)
    print(f"Saving to {filepath}")
    with open(filepath, "wb") as f:
        pickle.dump(np.array(grouped_shap_vals), f)
    print(f"Saving to {col_name_filepath}")
    with open(col_name_filepath, "wb") as f:
        pickle.dump(np.array(grouped_col_name_shap_vals), f)
    print(f"Saving to {colon_filepath}")
    with open(colon_filepath, "wb") as f:
        pickle.dump(np.array(grouped_colon_shap_vals), f)


if __name__ == "__main__":
    ds_type = parser.parse_args().ds_type
    sf = 1
    for model_type in [
        "ensemble_50",
        "ensemble_75",
        "ensemble_25",
        "stack",
        "all_text",
    ]:
        # pass
        shap_vals = run_shap(model_type, ds_type=ds_type, tab_scale_factor=sf)
    run_all_text_baseline_shap(ds_type=ds_type, tab_scale_factor=sf)
    gen_summary_shap_vals(ds_type, tab_scale_factor=sf)
