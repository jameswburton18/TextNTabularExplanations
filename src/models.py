import scipy.special
import numpy as np
import itertools
import shap
from datasets import load_dataset
from src.utils import MODEL_NAME_TO_DESC_DICT, format_text_pred, prepare_text
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForSequenceClassification, pipeline, AutoTokenizer
import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
from shap.maskers import Masker
from shap.utils.transformers import parse_prefix_suffix_for_tokenizer, getattr_silent
from shap.maskers._text import Token, TokenGroup, partition_tree, Text
import re

import lightgbm as lgb


class Model:
    def __init__(self, text_to_pred_dict=None, tab_model=None, text_pipeline=None):
        self.text_to_pred_dict = text_to_pred_dict
        self.tab_model = tab_model
        self.text_pipeline = text_pipeline

    def predict_both(self, examples, text_weight=0.5):
        """
        With a weighted ensemble like this, the text and tabular predictions are calculated separately,
        then combined. Therefore desc_dict is used to keep track of which examples have the same description
        and therefore the same text prediction and therefore only need to be predicted once.
        """
        if len(examples.shape) == 1:
            examples = examples.reshape(1, -1)
        tab_examples = examples[:, :-1]
        text_examples = examples[:, -1]
        tab_preds = self.tab_model.predict_proba(tab_examples)

        desc_dict = {}
        for i, desc in tqdm(enumerate(text_examples)):
            if desc not in desc_dict:
                desc_dict[desc] = [i]
            else:
                desc_dict[desc].append(i)

        if self.text_to_pred_dict is not None:
            text_preds = np.array(
                [self.text_to_pred_dict[desc] for desc in desc_dict.keys()]
            )

        else:
            dict_keys = list(desc_dict.keys())
            dict_keys = dict_keys[0] if len(dict_keys) == 1 else dict_keys
            text_preds = self.text_pipeline(dict_keys)
            text_preds = np.array([format_text_pred(pred) for pred in text_preds])

        expanded_text_preds = np.zeros((len(text_examples), 2))
        for i, (desc, idxs) in enumerate(desc_dict.items()):
            expanded_text_preds[idxs] = text_preds[i]

        # Combine the predictions, multiplying the text and predictions by 0.5
        preds = text_weight * expanded_text_preds + (1 - text_weight) * tab_preds
        return preds
