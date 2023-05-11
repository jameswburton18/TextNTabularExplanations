import numpy as np
from src.utils import format_text_pred
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm


class WeightedEnsemble:
    def __init__(
        self,
        tab_model,
        text_pipeline,
        cols_to_str_fn,
        text_weight=0.5,
    ):
        self.tab_model = tab_model
        self.text_pipeline = text_pipeline
        self.text_weight = text_weight
        self.cols_to_str_fn = cols_to_str_fn
        self.num_tab_cols = len(self.tab_model.feature_name_)

    def predict(self, examples):
        """
        With a weighted ensemble like this, the text and tabular predictions are calculated separately,
        then combined. Therefore desc_dict is used to keep track of which examples have the same description
        and therefore the same text prediction and therefore only need to be predicted once.
        """
        if len(examples.shape) == 1:
            examples = examples.reshape(1, -1)
        tab_examples = examples[:, : self.num_tab_cols]
        tab_preds = self.tab_model.predict_proba(tab_examples)
        text_examples = examples[:, self.num_tab_cols :]
        text_examples = np.array(list(map(self.cols_to_str_fn, text_examples)))

        desc_dict = {}

        for i, desc in tqdm(enumerate(text_examples)):
            if desc not in desc_dict:
                desc_dict[desc] = [i]
            else:
                desc_dict[desc].append(i)

        # Make a dictionary of the unique text examples and the indices of the examples that have that text
        dict_keys = list(desc_dict.keys())
        dict_keys = dict_keys[0] if len(dict_keys) == 1 else dict_keys
        text_preds = self.text_pipeline(dict_keys)
        text_preds = np.array([format_text_pred(pred) for pred in text_preds])

        expanded_text_preds = np.zeros((len(text_examples), text_preds.shape[1]))
        for i, (desc, idxs) in enumerate(desc_dict.items()):
            expanded_text_preds[idxs] = text_preds[i]

        # Combine the predictions, multiplying the text and predictions by 0.5
        preds = (
            self.text_weight * expanded_text_preds + (1 - self.text_weight) * tab_preds
        )
        return preds


class StackModel:
    def __init__(
        self,
        tab_model,
        text_pipeline,
        stack_model,
        cols_to_str_fn,
    ):
        self.tab_model = tab_model
        self.text_pipeline = text_pipeline
        self.stack_model = stack_model
        self.cols_to_str_fn = cols_to_str_fn
        self.num_tab_cols = len(self.tab_model.feature_name_)

    def predict(self, examples):
        if len(examples.shape) == 1:
            examples = examples.reshape(1, -1)
        tab_examples = examples[:, : self.num_tab_cols]
        text_examples = examples[:, self.num_tab_cols :]
        text_examples = np.array(list(map(self.cols_to_str_fn, text_examples)))
        tab_preds = self.tab_model.predict_proba(tab_examples)

        # As text and tabular are calucalted seperately, we can avoid recalculating text
        # predictions if we have already calculated them
        unique_texts = {}
        for i, desc in tqdm(enumerate(text_examples)):
            if desc not in unique_texts:
                unique_texts[desc] = [i]
            else:
                unique_texts[desc].append(i)

        dict_keys = list(unique_texts.keys())
        dict_keys = dict_keys[0] if len(dict_keys) == 1 else dict_keys
        text_preds = self.text_pipeline(dict_keys)
        text_preds = np.array([format_text_pred(pred) for pred in text_preds])

        expanded_text_preds = np.zeros((len(text_examples), text_preds.shape[1]))
        for i, (desc, idxs) in enumerate(unique_texts.items()):
            expanded_text_preds[idxs] = text_preds[i]

        # Stack
        stack_examples = np.hstack([tab_examples, expanded_text_preds, tab_preds])
        stack_preds = self.stack_model.predict_proba(stack_examples)

        return stack_preds


class AllAsTextModel:
    def __init__(self, text_pipeline, cols_to_str_fn):
        self.text_pipeline = text_pipeline
        self.cols_to_str_fn = cols_to_str_fn
        # self.cols = cols

    def predict(self, examples):
        # examples_as_strings = np.apply_along_axis(
        #     lambda x: array_to_string(x, self.cols), 1, examples
        # )
        examples_as_strings = np.array(list(map(self.cols_to_str_fn, examples)))
        preds = [
            out
            for out in self.text_pipeline(
                KeyDataset(Dataset.from_dict({"text": examples_as_strings}), "text"),
                batch_size=64,
            )
        ]
        preds = np.array([format_text_pred(pred) for pred in preds])
        # preds = np.array([[lab["score"] for lab in pred] for pred in preds])

        return preds
