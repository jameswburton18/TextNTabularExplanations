from datasets import Dataset, DatasetDict, load_dataset
from auto_mm_bench.datasets import dataset_registry
from sklearn.preprocessing import OrdinalEncoder
from src.dataset_info import get_dataset_info

# ## Dataset creation here
for dataset_name in [
    # "wine_reviews",
    # "fake_job_postings2",
    # "product_sentiment_machine_hack",
    # "kick_starter_funding",
    # "jigsaw_unintended_bias100K",
    # "imdb_genre_prediction",
    # "data_scientist_salary",
    # "melbourne_airbnb",
    "news_channel",
]:
    di = get_dataset_info(dataset_name)
    train_dataset = dataset_registry.create(dataset_name, "train")

    test_dataset = dataset_registry.create(dataset_name, "test")
    cols = train_dataset.feature_columns + train_dataset.label_columns

    train_txt = train_dataset.data[cols]
    test_txt = test_dataset.data[cols]

    # load dataset from dataframe
    train_ds = Dataset.from_pandas(train_txt)
    train_ds = train_ds.class_encode_column(train_dataset.label_columns[0])
    test_ds = Dataset.from_pandas(test_txt)
    test_ds = test_ds.class_encode_column(train_dataset.label_columns[0])

    train_ds = train_ds.train_test_split(
        test_size=0.15, seed=42, stratify_by_column=train_dataset.label_columns[0]
    )

    ds = DatasetDict(
        {"train": train_ds["train"], "validation": train_ds["test"], "test": test_ds}
    )

    # Now we have made the split but still need to deal with missing values, and that depends on the column type

    # All as text
    train_all_text = ds["train"].to_pandas()
    val_all_text = ds["validation"].to_pandas()
    test_all_text = ds["test"].to_pandas()

    train_all_text[train_dataset.feature_columns] = train_all_text[
        train_dataset.feature_columns
    ].astype("str")
    val_all_text[train_dataset.feature_columns] = val_all_text[
        train_dataset.feature_columns
    ].astype("str")
    test_all_text[train_dataset.feature_columns] = test_all_text[
        train_dataset.feature_columns
    ].astype("str")

    ds_all_text = DatasetDict(
        {
            "train": Dataset.from_pandas(train_all_text),
            "validation": Dataset.from_pandas(val_all_text),
            "test": Dataset.from_pandas(test_all_text),
        }
    )

    ds_all_text.push_to_hub(dataset_name + "_all_text")

    # Not all as text
    train = ds["train"].to_pandas()
    val = ds["validation"].to_pandas()
    test = ds["test"].to_pandas()

    train[di.text_cols] = train[di.text_cols].astype("str")
    val[di.text_cols] = val[di.text_cols].astype("str")
    test[di.text_cols] = test[di.text_cols].astype("str")

    # ds.push_to_hub(dataset_name)
    if len(di.categorical_cols) > 0:
        train[di.categorical_cols] = train[di.categorical_cols].astype("category")

        enc = OrdinalEncoder(encoded_missing_value=-1)
        train[di.categorical_cols] = enc.fit_transform(train[di.categorical_cols])

        val[di.categorical_cols] = val[di.categorical_cols].astype("category")
        val[di.categorical_cols] = enc.transform(val[di.categorical_cols])

        test[di.categorical_cols] = test[di.categorical_cols].astype("category")
        test[di.categorical_cols] = enc.transform(test[di.categorical_cols])

    ds2 = DatasetDict(
        {
            "train": Dataset.from_pandas(train),
            "validation": Dataset.from_pandas(val),
            "test": Dataset.from_pandas(test),
        }
    )

    ds2.push_to_hub(dataset_name + "_ordinal")
