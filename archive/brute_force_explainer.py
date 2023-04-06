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


# Now housed in models.py 

# class Model:
#     def __init__(self, text_to_pred_dict=None, tab_model=None, text_pipeline=None):
#         self.text_to_pred_dict = text_to_pred_dict
#         self.tab_model = tab_model
#         self.text_pipeline = text_pipeline

#     def predict_both(self, examples, text_weight=0.5):
#         if len(examples.shape) == 1:
#             examples = examples.reshape(1, -1)
#         tab_examples = examples[:, :-1]
#         text_examples = examples[:, -1]
#         tab_preds = self.tab_model.predict_proba(tab_examples)

#         desc_dict = {}
#         for i, desc in tqdm(enumerate(text_examples)):
#             if desc not in desc_dict:
#                 desc_dict[desc] = [i]
#             else:
#                 desc_dict[desc].append(i)

#         if self.text_to_pred_dict is not None:
#             text_preds = np.array(
#                 [self.text_to_pred_dict[desc] for desc in desc_dict.keys()]
#             )

#         else:
#             dict_keys = list(desc_dict.keys())
#             dict_keys = dict_keys[0] if len(dict_keys) == 1 else dict_keys
#             text_preds = self.text_pipeline(dict_keys)
#             text_preds = np.array([format_text_pred(pred) for pred in text_preds])

#         expanded_text_preds = np.zeros((len(text_examples), 2))
#         for i, (desc, idxs) in enumerate(desc_dict.items()):
#             expanded_text_preds[idxs] = text_preds[i]

#         # Combine the predictions, multiplying the text and predictions by 0.5
#         preds = text_weight * expanded_text_preds + (1 - text_weight) * tab_preds
#         return preds


def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


def shapley_kernel(M, s):
    if s == 0 or s == M:
        return 10000
    return (M - 1) / (scipy.special.binom(M, s) * s * (M - s))


def f(X):
    np.random.seed(0)
    beta = np.random.rand(X.shape[-1])
    return np.dot(X, beta) + 10


def kernel_shap(f, x, reference, M):
    X = np.zeros((2**M, M + 1))
    X[:, -1] = 1
    weights = np.zeros(2**M)
    V = np.zeros((2**M, M))
    for i in range(2**M):
        V[i, :] = reference

    ws = {}
    for i, s in enumerate(powerset(range(M))):
        s = list(s)
        V[i, s] = x[s]
        X[i, s] = 1
        ws[len(s)] = ws.get(len(s), 0) + shapley_kernel(M, len(s))
        weights[i] = shapley_kernel(M, len(s))
    y = f(V)
    tmp = np.linalg.inv(np.dot(np.dot(X.T, np.diag(weights)), X))
    return np.dot(tmp, np.dot(np.dot(X.T, np.diag(weights)), y))


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
text_masker = shap.maskers.Text(tokenizer)


def custom_masker(mask, x, M_tab):
    tab_mask = mask[:M_tab]
    # text_mask = np.concatenate(([1], mask[M_tab:], [1])) # add start and end tokens
    masked_tab = x[:M_tab] * tab_mask
    masked_text = text_masker(mask[M_tab:], x[M_tab])
    return np.hstack([masked_tab.astype("O"), masked_text[0]])


def my_kernel_shap(f, x, reference, M_tab, M_text):
    # M_text = M_text - 2 # for the start and end tokens
    M = M_tab + M_text
    X = np.zeros((2**M, M + 1))
    X[:, -1] = 1
    weights = np.zeros(2**M)
    V = np.zeros((2**M, M_tab + 1)).astype("O")

    ws = {}
    for i, s in enumerate(powerset(range(M))):
        s = list(s)
        X[i, s] = 1
        V[i] = custom_masker(X[i, :-1], x, M_tab)
        weights[i] = shapley_kernel(M, len(s))
    y = f(V, load_from_cache=False)
    tmp = np.linalg.inv(np.dot(np.dot(X.T, np.diag(weights)), X))
    return np.dot(tmp, np.dot(np.dot(X.T, np.diag(weights)), y))


def run_example():
    M = 4
    np.random.seed(1)
    x = np.random.randn(M)
    reference = np.zeros(M)
    phi = kernel_shap(f, x, reference, M)
    base_value = phi[-1]
    shap_values = phi[:-1]

    print("  reference =", reference)
    print("          x =", x)
    print("shap_values =", shap_values)
    print(" base_value =", base_value)
    print("   sum(phi) =", np.sum(phi))
    print("       f(x) =", f(x))


def run_proper():
    train_df = load_dataset(
        "james-burton/imdb_genre_prediction2", split="train[:10]"
    ).to_pandas()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    text_pipeline = pipeline(
        "text-classification",
        model="james-burton/imdb_genre_9",
        tokenizer=tokenizer,
        device="cuda:0",
    )
    # test_df = load_dataset('james-burton/imdb_genre_prediction2', split='test[:1]')
    tab_cols = [
        "Rating",
        "Votes",
    ]  # ['Year','Runtime (Minutes)', 'Rating', 'Votes', 'Revenue (Millions)','Metascore', 'Rank']
    text_col = ["Description"]

    # test_df_text = prepare_text(test_df, 'text_col_only')
    # test_df_tab = test_df.to_pandas()[tab_cols]

    train_df_tab = train_df[tab_cols]
    y_train = train_df["Genre_is_Drama"]

    tab_model = lgb.LGBMClassifier(random_state=42)
    tab_model.fit(train_df_tab, y_train)

    def tab_pred_fn(examples):
        preds = tab_model.predict_proba(examples)
        return preds

    test_model = Model(tab_model=tab_model, text_pipeline=text_pipeline)

    # We want to explain a single row
    np.random.seed(1)
    # x = np.array([['2009.0', '95.0', '7.7', '398972.0', '32.39', '76.0', '508.0',
    #     "An offbeat romantic comedy about a woman who doesn't believe true love exists, and the young man who falls for her."]])
    x = [7.7, 398972.0, "offbeat romantic comedy"]
    # M = len(tab_cols) + len(tokenizer(text_col))
    masker = shap.maskers.Text(tokenizer)
    mshape = masker.shape(x[-1])[1]
    M_tab = 2  # len(tab_cols)
    M_text = mshape
    tab_reference = shap.kmeans(train_df[tab_cols], 1).data
    phi = my_kernel_shap(test_model.predict_both, x, tab_reference, M_tab, M_text)
    base_value = phi[-1]
    shap_values = phi[:-1]

    print("  reference =", tab_reference)
    print("          x =", x)
    print("shap_values =", shap_values)
    print(" base_value =", base_value)
    print("   sum(phi) =", np.sum(phi, axis=0))
    print(
        "       f(x) =",
        test_model.predict_both(np.array([x], dtype="O"), load_from_cache=False),
    )


class JointMasker(Masker):
    def __init__(
        self,
        tokenizer=None,
        mask_token=None,
        collapse_mask_token="auto",
        output_type="string",
        tab_clustering=None,
        num_tab_features=None,
    ):
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.output_type = output_type
        self.collapse_mask_token = collapse_mask_token
        self.input_mask_token = mask_token
        self.mask_token = mask_token  # could be recomputed later in this function
        self.mask_token_id = mask_token if isinstance(mask_token, int) else None
        self.tab_clustering = tab_clustering
        parsed_tokenizer_dict = parse_prefix_suffix_for_tokenizer(self.tokenizer)

        self.keep_prefix = parsed_tokenizer_dict["keep_prefix"]
        self.keep_suffix = parsed_tokenizer_dict["keep_suffix"]

        self.text_data = True

        if mask_token is None:
            if getattr_silent(self.tokenizer, "mask_token") is not None:
                self.mask_token = self.tokenizer.mask_token
                self.mask_token_id = getattr_silent(self.tokenizer, "mask_token_id")
                if self.collapse_mask_token == "auto":
                    self.collapse_mask_token = False
            else:
                self.mask_token = "..."
        else:
            self.mask_token = mask_token

        if self.mask_token_id is None:
            self.mask_token_id = self.tokenizer(self.mask_token)["input_ids"][
                self.keep_prefix
            ]

        if self.collapse_mask_token == "auto":
            self.collapse_mask_token = True

        self.fixed_background = self.mask_token_id is None

        self.default_batch_size = 5

        # cache variables
        self._s = None
        self._tokenized_s_full = None
        self._tokenized_s = None
        self._segments_s = None

        # flag that we return outputs that will not get changed by later masking calls
        self.immutable_outputs = True

    def __call__(self, mask, x, M_tab):
        tab_mask = mask[:M_tab]
        # text_mask = np.concatenate(([1], mask[M_tab:], [1])) # add start and end tokens
        masked_tab = x[:M_tab] * tab_mask
        masked_text = text_masker(mask[M_tab:], x[M_tab])
        return np.hstack([masked_tab.astype("O"), masked_text[0]])

    def custom_clustering(self, s=[7.7, 398972.0, "offbeat romantic comedy"]):
        self._update_s_cache(s)
        special_tokens = []
        sep_token = getattr_silent(self.tokenizer, "sep_token")
        if sep_token is None:
            special_tokens = []
        else:
            special_tokens = [sep_token]

        # convert the text segments to tokens that the partition tree function expects
        tokens = []
        space_end = re.compile(r"^.*\W$")
        letter_start = re.compile(r"^[A-z]")
        for i, v in enumerate(self._segments_s):
            if (
                i > 0
                and space_end.match(self._segments_s[i - 1]) is None
                and letter_start.match(v) is not None
                and tokens[i - 1] != ""
            ):
                tokens.append("##" + v.strip())
            else:
                tokens.append(v.strip())

        text_pt = partition_tree(tokens, special_tokens)
        tab_pt = self.tab_clustering

        """
        Dendrograms creation works by having each one of the base leaves as a number, then
        labelling each one of the new created nodes a number following the last leaf number.
        
        eg for array([[0. , 1. , 0.4, 2. ],
        [2. , 3. , 0.4, 2. ],
        [6. , 4. , 0.6, 3. ],
        [5. , 7. , 1. , 5. ]])
        
        In this case we know previously leaves are [0,1,2,3,4] (I don't think there is an easy
        way to calculate this from the dendrogram itself). Therefore the pairing of (0,1) from
        row 0 is labelled as 5, (2,3) from row 1 is labelled as 6, (6,4) from row 2 is labelled as 7
        and (5,7) from row 3 is labelled as 8.
        """
        n_text_groups = len(text_pt)
        n_tab_groups = len(tab_pt)

        # References to non-leaf nodes need to be shifted by the number of new leaves
        Z_join = np.zeros([n_tab_groups + n_text_groups + 1, 4])
        Z_join[:n_text_groups, :2] = np.where(
            text_pt[:, :2] >= n_text_leaves,
            text_pt[:, :2] + n_tab_leaves,
            text_pt[:, :2],
        )
        Z_join[n_text_groups:-1, :2] = np.where(
            tab_pt[:, :2] >= n_tab_leaves,
            tab_pt[:, :2] + n_text_leaves + n_text_groups,
            tab_pt[:, :2] + n_text_leaves,
        )
        # 3rd and 4th columns are left unchanged
        Z_join[:n_text_groups, 2:] = text_pt[:, 2:]
        Z_join[n_text_groups:-1, 2:] = tab_pt[:, 2:]

        # Z_text[:,:2] += (Z_text[:,:2]>=n_text_leaves)*n_tab_leaves
        # Z_tab[:,:2] = np.where(Z_tab[:,:2]>=n_tab_leaves,Z_tab[:,:2]+n_text_leaves+ n_text_groups,
        #                        Z_tab[:,:2]+n_text_leaves)

        # Create top join, joining the text and tab dendrograms together
        top_text_node = n_text_leaves + n_tab_leaves + n_text_groups + -1
        top_tab_node = top_text_node + n_tab_groups
        # Set similarity of top node to 1.5
        Z_join[-1, :] = np.array(
            [top_text_node, top_tab_node, 1.5, n_tab_leaves + n_text_leaves]
        )

        return Z_join


openers = {"(": ")"}
closers = {")": "("}
enders = [".", ","]
connectors = ["but", "and", "or"]


def merge_score(group1, group2, special_tokens):
    """Compute the score of merging two token groups.

    special_tokens: tokens (such as separator tokens) that should be grouped last
    """
    score = 0
    # ensures special tokens are combined last, so 1st subtree is 1st sentence and 2nd subtree is 2nd sentence
    if len(special_tokens) > 0:
        if group1[-1].s in special_tokens and group2[0].s in special_tokens:
            score -= (
                math.inf
            )  # subtracting infinity to create lowest score and ensure combining these groups last

    # merge broken-up parts of words first
    if group2[0].s.startswith("##"):
        score += 20

    # merge apostrophe endings next
    if group2[0].s == "'" and (
        len(group2) == 1 or (len(group2) == 2 and group2[1].s in ["t", "s"])
    ):
        score += 15
    if group1[-1].s == "'" and group2[0].s in ["t", "s"]:
        score += 15

    start_ctrl = group1[0].s.startswith("[") and group1[0].s.endswith("]")
    end_ctrl = group2[-1].s.startswith("[") and group2[-1].s.endswith("]")

    if (start_ctrl and not end_ctrl) or (end_ctrl and not start_ctrl):
        score -= 1000
    if group2[0].s in openers and not group2[0].balanced:
        score -= 100
    if group1[-1].s in closers and not group1[-1].balanced:
        score -= 100

    # attach surrounding an openers and closers a bit later
    if group1[0].s in openers and not group2[-1] in closers:
        score -= 2

    # reach across connectors later
    if group1[-1].s in connectors or group2[0].s in connectors:
        score -= 2

    # reach across commas later
    if group1[-1].s == ",":
        score -= 10
    if group2[0].s == ",":
        if len(group2) > 1:  # reach across
            score -= 10
        else:
            score -= 1

    # reach across sentence endings later
    if group1[-1].s in [".", "?", "!"]:
        score -= 20
    if group2[0].s in [".", "?", "!"]:
        if len(group2) > 1:  # reach across
            score -= 20
        else:
            score -= 1

    score -= len(group1) + len(group2)
    # print(group1, group2, score)
    return score


def merge_closest_groups(groups, special_tokens):
    """Finds the two token groups with the best merge score and merges them."""
    scores = [
        merge_score(groups[i], groups[i + 1], special_tokens)
        for i in range(len(groups) - 1)
    ]
    # print(scores)
    ind = np.argmax(scores)
    groups[ind] = groups[ind] + groups[ind + 1]
    # print(groups[ind][0].s in openers, groups[ind][0])
    if (
        groups[ind][0].s in openers
        and groups[ind + 1][-1].s == openers[groups[ind][0].s]
    ):
        groups[ind][0].balanced = True
        groups[ind + 1][-1].balanced = True

    groups.pop(ind + 1)


def text_partition_tree(decoded_tokens, special_tokens):
    """Build a heriarchial clustering of tokens that align with sentence structure.

    Note that this is fast and heuristic right now.
    TODO: Build this using a real constituency parser.
    """
    token_groups = [TokenGroup([Token(t)], i) for i, t in enumerate(decoded_tokens)]
    #     print(token_groups)
    M = len(decoded_tokens)
    new_index = M
    clustm = np.zeros((M - 1, 4))
    for i in range(len(token_groups) - 1):
        scores = [
            merge_score(token_groups[i], token_groups[i + 1], special_tokens)
            for i in range(len(token_groups) - 1)
        ]
        #         print(scores)
        ind = np.argmax(scores)

        lind = token_groups[ind].index
        rind = token_groups[ind + 1].index
        clustm[new_index - M, 0] = token_groups[ind].index
        clustm[new_index - M, 1] = token_groups[ind + 1].index
        clustm[new_index - M, 2] = -scores[ind]
        clustm[new_index - M, 3] = (clustm[lind - M, 3] if lind >= M else 1) + (
            clustm[rind - M, 3] if rind >= M else 1
        )

        token_groups[ind] = token_groups[ind] + token_groups[ind + 1]
        token_groups[ind].index = new_index

        # track balancing of openers/closers
        if (
            token_groups[ind][0].s in openers
            and token_groups[ind + 1][-1].s == openers[token_groups[ind][0].s]
        ):
            token_groups[ind][0].balanced = True
            token_groups[ind + 1][-1].balanced = True

        token_groups.pop(ind + 1)
        new_index += 1

    # negative means we should never split a group, so we add 10 to ensure these are very tight groups
    # (such as parts of the same word)
    clustm[:, 2] = clustm[:, 2] + 10

    return clustm


if __name__ == "__main__":
    # run_example()
    # run_proper()
    masker = JointMasker()
    sample_text = "organic romantic comedy"
    masker.custom_clustering(sample_text)
