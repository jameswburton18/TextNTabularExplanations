import numpy as np
import shap
from shap.maskers import Masker

# from shap.maskers._text import partition_tree, Text, SimpleTokenizer
from shap.maskers._tabular import Tabular, _delta_masking
from shap.utils import safe_isinstance, MaskedModel, sample
from shap.utils.transformers import (
    parse_prefix_suffix_for_tokenizer,
    SENTENCEPIECE_TOKENIZERS,
    getattr_silent,
)
from shap.utils._exceptions import DimensionError, InvalidClusteringError
from datasets import load_dataset
from src.utils import format_text_pred
from transformers import pipeline, AutoTokenizer
import pandas as pd
from datasets import load_dataset, Dataset
from transformers.pipelines.pt_utils import KeyDataset
import re
import scipy as sp

# from src.models import Model
import lightgbm as lgb
from src.models import WeightedEnsemble
import math

# import faulthandler

# faulthandler.enable()


class JointMasker(Masker):
    def __init__(
        self,
        tab_df,
        text_cols,
        cols_to_str_fn,
        max_samples=100,
        tokenizer=None,
        mask_token=None,
        collapse_mask_token="auto",
        output_type="string",
        tab_cluster_scale_factor=2,
        tab_partition_tree=None,
    ):
        if tokenizer is None:
            self.tokenizer = SimpleTokenizer()
        elif callable(tokenizer):
            self.tokenizer = tokenizer
        else:
            try:
                self.tokenizer = SimpleTokenizer(tokenizer)
            except:
                raise Exception(  # pylint: disable=raise-missing-from
                    "The passed tokenizer cannot be wrapped as a masker because it does not have a __call__ "
                    + "method, not can it be interpreted as a splitting regexp!"
                )
        self.output_type = output_type
        self.collapse_mask_token = collapse_mask_token
        self.input_mask_token = mask_token
        self.mask_token = mask_token  # could be recomputed later in this function
        self.mask_token_id = mask_token if isinstance(mask_token, int) else None
        self.tab_cluster_scale_factor = (
            tab_cluster_scale_factor  # This could be important
        )

        # Tab
        self.output_dataframe = False
        if safe_isinstance(tab_df, "pandas.core.frame.DataFrame"):
            self.tab_feature_names = list(tab_df.columns)
            tab_df = tab_df.values
            self.output_dataframe = True

        # Tab clustering
        self.n_tab_cols = tab_df.shape[-1]
        # In order to cluster the tabular data, we replace null values with median
        if tab_partition_tree is None:
            self.tab_pt = sp.cluster.hierarchy.complete(
                sp.spatial.distance.pdist(
                    pd.DataFrame(tab_df).fillna(pd.DataFrame(tab_df).median()).values.T,
                    metric="correlation",
                )
            )
        else:
            self.tab_pt = tab_partition_tree
        self.n_tab_groups = len(self.tab_pt)

        if hasattr(tab_df, "shape") and tab_df.shape[0] > max_samples:
            tab_df = sample(tab_df, max_samples)

        self.data = tab_df
        self.max_samples = max_samples

        self._masked_data = tab_df.copy()
        self._last_mask = np.zeros(tab_df.shape[1], dtype="bool")
        self.tab_shape = tab_df.shape
        self.supports_delta_masking = True

        # Text
        self.text_cols = text_cols
        self.cols_to_text_fn = cols_to_str_fn
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

    def __call__(self, mask, x):
        """
        tab_mask_call returns a dataframe of shape (num_samples, num_tab_features)
        text_mask_call returns a tuple of an array of a string, with the mask applied to the text
        However, if only the text is being masked, then we do not need to sample
        from tab_df and can just take the first row
        """
        masked_tab = self.tab_mask_call(mask[: self.n_tab_cols], x[: self.n_tab_cols])

        self._text_ft_index_ends(x[self.n_tab_cols :])
        # We join the text cols into a single string. This is independent of how they are joined in the model,
        # as we are only interested in the tokenization for the masking
        text = " ".join(str(s) for s in x[self.n_tab_cols :])
        masked_text = self.text_mask_call(mask[self.n_tab_cols :], text)

        # We unpack the string from the tuple and array
        masked_tab[self.text_cols] = masked_text[0][0]

        return masked_tab.values

    def _text_ft_index_ends(self, s):
        lens = []
        sent_indices = []
        for idx, col in enumerate(s):
            # First text col
            if lens == []:
                tokens, token_ids = self.token_segments(str(col))
                # -1 as we don't use SEP tokens (unless it's the only text col)
                also_last = 1 if len(s) == 1 else 0
                token_len = len(tokens) - 1 + also_last
                lens.append(token_len - 1)
                sent_indices.extend([idx] * token_len)
            # Last text col
            elif idx == len(s) - 1:
                tokens, token_ids = self.token_segments(str(col))
                # -1 for CLS tokens
                token_len = len(tokens) - 1
                lens.append(lens[-1] + token_len)
                sent_indices.extend([idx] * token_len)
            # Middle text cols
            else:
                tokens, token_ids = self.token_segments(str(col))
                # -2 for CLS and SEP tokens
                token_len = len(tokens) - 2
                lens.append(lens[-1] + token_len)
                sent_indices.extend([idx] * token_len)

        self._sent_split_idxs = lens[:-1]
        self.sent_indices = sent_indices

    def tab_mask_call(self, mask, x):
        mask = self._standardize_mask(mask, x)

        # make sure we are given a single sample
        if len(x.shape) != 1 or x.shape[0] != self.data.shape[1]:
            raise DimensionError(
                "The input passed for tabular masking does not match the background data shape!"
            )

        # if mask is an array of integers then we are doing delta masking
        if np.issubdtype(mask.dtype, np.integer):
            variants = ~self.invariants(x)
            curr_delta_inds = np.zeros(len(mask), dtype=np.int)
            num_masks = (mask >= 0).sum()
            varying_rows_out = np.zeros((num_masks, self.tab_shape[0]), dtype="bool")
            masked_inputs_out = np.zeros(
                (num_masks * self.tab_shape[0], self.tab_shape[1])
            )
            self._last_mask[:] = False
            self._masked_data[:] = self.data
            _delta_masking(
                mask,
                x,
                curr_delta_inds,
                varying_rows_out,
                self._masked_data,
                self._last_mask,
                self.data,
                variants,
                masked_inputs_out,
                MaskedModel.delta_mask_noop_value,
            )
            if self.output_dataframe:
                return (
                    pd.DataFrame(masked_inputs_out, columns=self.tab_feature_names),
                ), varying_rows_out

            return (masked_inputs_out,), varying_rows_out

        # otherwise we update the whole set of masked data for a single sample
        self._masked_data[:] = x * mask + self.data * np.invert(mask)
        self._last_mask[:] = mask

        if self.output_dataframe:
            return pd.DataFrame(self._masked_data, columns=self.tab_feature_names)

        return (self._masked_data,)

    def text_mask_call(self, mask, s):
        # text = " ".join(s[self.n_tab_cols :])
        text = s
        mask = self._standardize_mask(mask, text)
        self._update_s_cache(text)

        # if we have a fixed prefix or suffix then we need to grow the mask to account for that
        if self.keep_prefix > 0 or self.keep_suffix > 0:
            mask = mask.copy()
            mask[: self.keep_prefix] = True
            mask[-self.keep_suffix :] = True

        if self.output_type == "string":
            out_parts = []
            out = []
            is_previous_appended_token_mask_token = False
            sep_token = getattr_silent(self.tokenizer, "sep_token")
            for i, v in enumerate(mask):
                # mask ignores separator tokens and keeps them unmasked
                if v or sep_token == self._segments_s[i]:
                    out_parts.append(self._segments_s[i])
                    is_previous_appended_token_mask_token = False
                    # Change in here to show diffs between desc and title
                else:
                    # If we don't collapse any mask tokens then we add another mask token
                    # Or if the previous appended token was not a mask token
                    if not self.collapse_mask_token or (
                        self.collapse_mask_token
                        and not is_previous_appended_token_mask_token
                    ):
                        out_parts.append(" " + self.mask_token)
                        is_previous_appended_token_mask_token = True
                        # Length 9, 0,1,2,3,4 is the first group, 5,6 2nd and 7,8 3rd
                        # All the masks are false, so
                if i in self._sent_split_idxs:
                    out.append(" ".join(out_parts))
                    out_parts = []
                    is_previous_appended_token_mask_token = False
            out.append(" ".join(out_parts))

            for i in range(len(out)):
                out[i] = re.sub(r"[\s]+", " ", out[i]).strip()
                if safe_isinstance(self.tokenizer, SENTENCEPIECE_TOKENIZERS):
                    out[i] = out[i].replace("▁", " ")

        else:
            if self.mask_token_id is None:
                out = self._tokenized_s[mask]
            else:
                out = np.array(
                    [
                        self._tokenized_s[i] if mask[i] else self.mask_token_id
                        for i in range(len(mask))
                    ]
                )

        # for some sentences with strange configurations around the separator tokens, tokenizer encoding/decoding may contain
        # extra unnecessary tokens, for example ''. you may want to strip out spaces adjacent to separator tokens. Refer to PR
        # for more details.
        return (np.array([out]),)

    def _update_s_cache(self, s):
        """Same as Text masker"""
        if self._s != s:
            self._s = s
            tokens, token_ids = self.token_segments(s)
            self._tokenized_s = np.array(token_ids)
            self._segments_s = np.array(tokens)

    def token_segments(self, s):
        """Same as Text masker"""
        """ Returns the substrings associated with each token in the given string.
        """

        try:
            token_data = self.tokenizer(s, return_offsets_mapping=True)
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
            token_ids = self.tokenizer(s)["input_ids"]
            if hasattr(self.tokenizer, "convert_ids_to_tokens"):
                tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            else:
                tokens = [self.tokenizer.decode([id]) for id in token_ids]
            if hasattr(self.tokenizer, "get_special_tokens_mask"):
                special_tokens_mask = self.tokenizer.get_special_tokens_mask(
                    token_ids, already_has_special_tokens=True
                )
                # avoid masking separator tokens, but still mask beginning of sentence and end of sentence tokens
                special_keep = [
                    getattr_silent(self.tokenizer, "sep_token"),
                    getattr_silent(self.tokenizer, "mask_token"),
                ]
                for i, v in enumerate(special_tokens_mask):
                    if v == 1 and (
                        tokens[i] not in special_keep
                        or i + 1 == len(special_tokens_mask)
                    ):
                        tokens[i] = ""

            # add spaces to separate the tokens (since we want segments not tokens)
            if safe_isinstance(self.tokenizer, SENTENCEPIECE_TOKENIZERS):
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

    def clustering(self, s):
        text = " ".join(str(s) for s in s[self.n_tab_cols :])
        # text = self.cols_to_text_fn(s[self.n_tab_cols :])
        self._update_s_cache(text)
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

        text_pt = partition_tree(tokens, special_tokens, self.sent_indices)
        text_pt[:, 2] = text_pt[:, 3]
        text_pt[:, 2] /= text_pt[:, 2].max()

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
        n_text_leaves = len(tokens)
        n_text_groups = len(text_pt)

        # References to non-leaf nodes need to be shifted by the number of new leaves
        Z_join = np.zeros([self.n_tab_groups + n_text_groups + 1, 4])

        # Put tab first, then text
        Z_join[: self.n_tab_groups, :2] = np.where(
            self.tab_pt[:, :2] >= self.n_tab_cols,
            self.tab_pt[:, :2] + n_text_leaves,
            self.tab_pt[:, :2],
        )
        Z_join[self.n_tab_groups : -1, :2] = np.where(
            text_pt[:, :2] >= n_text_leaves,
            text_pt[:, :2] + self.n_tab_cols + self.n_tab_groups,
            text_pt[:, :2] + self.n_tab_cols,
        )

        # Scale tab_pt
        self.tab_pt[:, 2] /= self.tab_pt[:, 2].max() * self.tab_cluster_scale_factor

        # 3rd and 4th columns are left unchanged
        Z_join[: self.n_tab_groups, 2:] = self.tab_pt[:, 2:]
        Z_join[self.n_tab_groups : -1, 2:] = text_pt[:, 2:]

        # Create top join, joining the text and tab dendrograms together
        top_tab_node = self.n_tab_cols + n_text_leaves + self.n_tab_groups - 1
        top_text_node = top_tab_node + n_text_groups
        # Set similarity of top node to 1.2
        Z_join[-1, :] = np.array(
            [top_tab_node, top_text_node, 1.2, n_text_leaves + self.n_tab_cols]
        )

        return Z_join

    def shape(self, s):
        """The shape of what we return as a masker.

        Note we only return a single sample, so there is no expectation averaging.
        """
        text = " ".join(str(s) for s in s[self.n_tab_cols :])
        # text = self.cols_to_text_fn(s[self.n_tab_cols :])
        self._update_s_cache(text)
        return (self.max_samples, self.n_tab_cols + len(self._tokenized_s))

    def mask_shapes(self, s):
        """The shape of the masks we expect."""
        text = " ".join(str(s) for s in s[self.n_tab_cols :])
        # text = self.cols_to_text_fn(s[self.n_tab_cols :])
        self._update_s_cache(text)
        return [(self.n_tab_cols + len(self._tokenized_s),)]

    def feature_names(self, s):
        """The names of the features for each mask position for the given input string."""
        text = " ".join(str(s) for s in s[self.n_tab_cols :])
        # text = self.cols_to_text_fn(s[self.n_tab_cols :])
        self._update_s_cache(text)
        return [self.tab_feature_names + [v.strip() for v in self._segments_s]]


class SimpleTokenizer:  # pylint: disable=too-few-public-methods
    """A basic model agnostic tokenizer."""

    def __init__(self, split_pattern=r"\W+"):
        """Create a tokenizer based on a simple splitting pattern."""
        self.split_pattern = re.compile(split_pattern)

    def __call__(self, s, return_offsets_mapping=True):
        """Tokenize the passed string, optionally returning the offsets of each token in the original string."""
        pos = 0
        offset_ranges = []
        input_ids = []
        for m in re.finditer(self.split_pattern, s):
            start, end = m.span(0)
            offset_ranges.append((pos, start))
            input_ids.append(s[pos:start])
            pos = end
        if pos != len(s):
            offset_ranges.append((pos, len(s)))
            input_ids.append(s[pos:])

        out = {}
        out["input_ids"] = input_ids
        if return_offsets_mapping:
            out["offset_mapping"] = offset_ranges
        return out


def post_process_sentencepiece_tokenizer_output(s):
    """replaces whitespace encoded as '_' with ' ' for sentencepiece tokenizers."""
    s = s.replace("▁", " ")
    return s


openers = {"(": ")"}
closers = {")": "("}
enders = [".", ","]
connectors = ["but", "and", "or"]


class Token:
    """A token representation used for token clustering."""

    def __init__(self, value, sent_no):
        self.s = value
        self.sent_no = sent_no
        if value in openers or value in closers:
            self.balanced = False
        else:
            self.balanced = True

    def __str__(self):
        return self.s

    def __repr__(self):
        if not self.balanced:
            return self.s + "!"
        return self.s


class TokenGroup:
    """A token group (substring) representation used for token clustering."""

    def __init__(self, group, index=None):
        self.g = group
        self.index = index

    def __repr__(self):
        return self.g.__repr__()

    def __getitem__(self, index):
        return self.g[index]

    def __add__(self, o):
        return TokenGroup(self.g + o.g)

    def __len__(self):
        return len(self.g)


def merge_score(group1, group2, special_tokens):
    """Compute the score of merging two token groups.

    special_tokens: tokens (such as separator tokens) that should be grouped last
    """
    # if type(group1)

    score = 0

    # Changed
    ########################################
    if group1[-1].sent_no != group2[0].sent_no:
        score -= (
            math.inf
        )  # subtracting infinity to create lowest score and ensure combining these groups last

    ########################################

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


def partition_tree(decoded_tokens, special_tokens, sent_indices):
    """Build a heriarchial clustering of tokens that align with sentence structure.

    Note that this is fast and heuristic right now.
    TODO: Build this using a real constituency parser.
    """
    token_groups = [
        TokenGroup([Token(t, idx)], i)
        for i, (t, idx) in enumerate(zip(decoded_tokens, sent_indices))
    ]
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
    # run_shap_vals('tab')
    # run_shap_vals("joint")
    pass
