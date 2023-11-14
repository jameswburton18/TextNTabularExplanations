import numpy as np
from shap.maskers import Masker
from shap.maskers._tabular import _delta_masking
from shap.maskers._text import (
    SimpleTokenizer,
    openers,
    closers,
    connectors,
    TokenGroup,
)
import shap
from shap.utils import safe_isinstance, MaskedModel, sample
from shap.utils.transformers import (
    parse_prefix_suffix_for_tokenizer,
    SENTENCEPIECE_TOKENIZERS,
    getattr_silent,
)
from shap.utils._exceptions import DimensionError
import pandas as pd
import re
import scipy as sp
import math


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
        tab_cluster_scale_factor=1,
        tab_partition_tree=None,
    ):
        # Boiler plate from Text masker
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
        # Optional scaling
        self.tab_cluster_scale_factor = tab_cluster_scale_factor

        # Tab
        self.output_dataframe = False
        if safe_isinstance(tab_df, "pandas.core.frame.DataFrame"):
            self.tab_feature_names = list(tab_df.columns)
            tab_df = tab_df.values
            self.output_dataframe = True

        # Tab clustering, partition tree calculated as in Tabular masker (correlation)
        self.n_tab_cols = tab_df.shape[-1]
        if tab_df.shape[-1] > 1:
            # In order to cluster the tabular data, we replace null values with median
            if tab_partition_tree is None:
                self.tab_pt = sp.cluster.hierarchy.complete(
                    sp.spatial.distance.pdist(
                        pd.DataFrame(tab_df)
                        .fillna(pd.DataFrame(tab_df).median())
                        .values.T,
                        metric="correlation",
                    )
                )
            else:
                self.tab_pt = tab_partition_tree
            # Necessary for adjusting text partition tree
            self.n_tab_groups = len(self.tab_pt)
        else:
            # Needed to add this in to avoid errors when there is only one tabular column
            self.tab_pt = None
            self.n_tab_groups = 1

        # Background dataset for tabular data set at max_samples
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
        We join the text cols into a single string. This is independent of how they are joined in the model,
        as we are only interested in the tokenization for the masking. This does mean that it won't work
        for models that tokenize spaces (I think).

        Possibly could do it instead by tokenizing all text features seperately, then joining them together
        and handling the start and end tokens seperately.
        """
        masked_tab = self.tab_mask_call(mask[: self.n_tab_cols], x[: self.n_tab_cols])
        masked_text = self.text_mask_call(mask[self.n_tab_cols :], x[self.n_tab_cols :])

        # We unpack the string from the tuple and array and extend the masked_tab dataframe
        masked_tab[self.text_cols] = masked_text[0]

        return masked_tab.values

    def tab_mask_call(self, mask, x):
        """
        Taken from Tabular masker, with little change
        """
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
        """
        Taken from text masker, changed to ensure that the mask tokens are not collapsed across
        different text features
        """
        mask = self._standardize_mask(mask, s)
        self._update_s_cache(s)

        # if we have a fixed prefix or suffix then we need to grow the mask to account for that
        if self.keep_prefix > 0 or self.keep_suffix > 0:
            mask = mask.copy()
            mask[: self.keep_prefix] = True
            mask[-self.keep_suffix :] = True
        #  split mask into groups based on [len(col) for col in self._tokenized_s] ([9, 11, 12])
        mask_per_col = np.split(
            mask, np.cumsum([len(col) for col in self._tokenized_s])[:-1]
        )
        out = []
        for col_idx, mask in enumerate(mask_per_col):
            if self.output_type == "string":
                out_parts = []
                col_out = []
                is_previous_appended_token_mask_token = False
                sep_token = getattr_silent(self.tokenizer, "sep_token")
                for i, v in enumerate(mask):
                    # mask ignores separator tokens and keeps them unmasked
                    if v or sep_token == self._segments_s[col_idx][i]:
                        out_parts.append(self._segments_s[col_idx][i])
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
                    # if i in self._sent_split_idxs:
                    #     out.append("".join(out_parts))
                    #     out_parts = []
                    #     is_previous_appended_token_mask_token = False
                col_out.append("".join(out_parts))
                # out.append("".join("".join(out_parts).split(self.tokenizer.sep_token)))

                for i in range(len(col_out)):
                    col_out[i] = re.sub(r"[\s]+", " ", col_out[i]).strip()
                    if safe_isinstance(self.tokenizer, SENTENCEPIECE_TOKENIZERS):
                        col_out[i] = col_out[i].replace("▁", " ")

            else:
                if self.mask_token_id is None:
                    col_out = self._tokenized_s[col_idx][mask]
                else:
                    col_out = np.array(
                        [
                            self._tokenized_s[col_idx][i]
                            if mask[i]
                            else self.mask_token_id
                            for i in range(len(mask))
                        ]
                    )
            out.extend(col_out)
        # for some sentences with strange configurations around the separator tokens, tokenizer encoding/decoding may contain
        # extra unnecessary tokens, for example ''. you may want to strip out spaces adjacent to separator tokens. Refer to PR
        # for more details.
        # print(out)
        assert len(out) == len(self._tokenized_s)
        # return
        return (np.array(out),)

    def _update_s_cache(self, s):
        # """Same as Text masker"""
        joined_s = "".join(col for col in s)
        if self._s != joined_s:
            self._s = joined_s
            if len(s) == 1:
                tokens, token_ids = self.token_segments(s[0])
                self._tokenized_s = [np.array(token_ids)]
                self._segments_s = [np.array(tokens)]
            else:
                all_tokens = []
                all_token_ids = []
                for col_idx, text_col in enumerate(s):
                    col_tokens, col_token_ids = self.token_segments(text_col)
                    """
                    We differentiate these because the first column we need to get 
                    rid of the end of sentence token, the last column we need to get 
                    rid of the start of sentence token and the middle columns we need
                    to get rid of both.
                    
                    We also need to make allowances for when the tokenizer doesn't have
                    a start or end of sentence token.
                    """
                    # First col
                    if col_idx == 0:
                        col_token_ids = (
                            col_token_ids[:-1] if self.keep_suffix else col_token_ids
                        )
                        col_tokens = col_tokens[:-1] if self.keep_suffix else col_tokens
                    # Middle cols
                    elif col_idx != len(s) - 1:
                        col_token_ids = (
                            col_token_ids[self.keep_prefix : -1]
                            if self.keep_suffix
                            else col_token_ids[self.keep_prefix :]
                        )
                        col_tokens = (
                            col_tokens[self.keep_prefix : -1]
                            if self.keep_suffix
                            else col_tokens[self.keep_prefix :]
                        )
                    # Last col
                    else:
                        col_token_ids = col_token_ids[self.keep_prefix :]
                        col_tokens = col_tokens[self.keep_prefix :]
                    all_tokens.append(np.array(col_tokens))
                    all_token_ids.append(np.array(col_token_ids))
                self._tokenized_s = all_token_ids
                self._segments_s = all_tokens

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
        """
        [Dendograms explained:]

        Dendrograms creation works by having each one of the base leaves as a number, then
        labelling each one of the new created nodes a number following the last leaf number.
        Columns 0 and 1 are the two nodes that are being joined, column 2 is the similarity
        between the two nodes and column 3 is the number of leaves in the new node.

        eg for array(
            [[0. , 1. , 0.4, 2. ],
            [2. , 3. , 0.4, 2. ],
            [6. , 4. , 0.6, 3. ],
            [5. , 7. , 1. , 5. ]]
        )

        In this case we know can see from the 4th row that there are 5 leaves in the final node:
        [0,1,2,3,4]. Each grouping (row) is also given a number starting from one more than the
        last leaf node. Therefore the pairing of (0,1) from row 0 is labelled as 5, (2,3) from
        row 1 is labelled as 6, (6,4) from row 2 is labelled as 7 and (5,7) from row 3 is labelled
        as 8. This tells the algorithm how to plot the dendrogram. For example row 3 tells us that
        the group of (2,3) ie node 6 is joined to node 4, which is the group of (6,4) ie node 7.

        With this knowledge we can adjust the text dendrogram in order to join the tabular dendrogram
        on from the left side.

        If we initially have:
            A tabular dendogram which has:
                * n leaves (labelled [0,n-1]
                * m groups (labelled [n, n+m-1])
            A text dendrogram which has:
                * k leaves (labelled [0,k-1]
                * l groups (labelled [k, k+l-1])

        This will now become a joint dendogram which has :
        * n tabular leaves (labelled [0,n-1]
        * k text leaves (labelled [n, n+k-1])
        * m tabular groups (labelled [n+k, n+k+m-1])
        * l text groups (labelled [n+k+m, n+k+m+l-1])

        """
        # text = " ".join(str(s) for s in x[self.n_tab_cols :])
        # text = self.tokenizer.sep_token.join(str(s) for s in s[self.n_tab_cols :])
        text = s[self.n_tab_cols :]
        # joiner = getattr_silent(self.tokenizer, "sep_token")
        # joiner = " " if joiner is None else joiner
        # text = joiner.join(str(s) for s in s[self.n_tab_cols :])

        # Same as Text masker
        ################################
        self._update_s_cache(text)
        special_tokens = []
        sep_token = getattr_silent(self.tokenizer, "sep_token")
        if sep_token is None:
            special_tokens = []
        else:
            special_tokens = [sep_token]

        text_pts = []
        all_tokens = []
        for col in range(len(self._tokenized_s)):
            # convert the text segments to tokens that the partition tree function expects
            tokens = []
            space_end = re.compile(r"^.*\W$")
            letter_start = re.compile(r"^[A-z]")
            for i, v in enumerate(self._segments_s[col]):
                if (
                    i > 0
                    and space_end.match(self._segments_s[col][i - 1]) is None
                    and letter_start.match(v) is not None
                    and tokens[i - 1] != ""
                ):
                    tokens.append("##" + v.strip())
                else:
                    tokens.append(v.strip())

            # text_pt = partition_tree(tokens, special_tokens, self.sent_indices)
            text_pt = shap.maskers._text.partition_tree(tokens, special_tokens)
            if len(text_pt) == 0:
                text_pt = np.array([[0, np.inf, 0, 1]])
                # text_pt[:, 2] = text_pt[:, 3]
                # text_pt[:, 2] /= text_pt[:, 2].max()
            text_pts.append(text_pt)
            all_tokens.extend(tokens)
        ################################

        if len(text_pts) == 1:
            text_pt = text_pts[0]
        else:
            text_pt = join_dendograms(text_pts)
        n_text_leaves = len(all_tokens)
        n_text_groups = len(text_pt)

        if self.tab_pt is not None:
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
        else:
            # If there is no tab_pt then this means there is only one tab column. In the resulting dendogram,
            # the single tab column will be joined to the text dendrogram on the left
            Z_join = np.zeros([n_text_groups + self.n_tab_cols, 4])
            Z_join[:-1, :2] = np.where(
                text_pt[:, :2] >= n_text_leaves,
                text_pt[:, :2] + self.n_tab_cols,
                text_pt[:, :2] + self.n_tab_cols,
            )
            # 3rd and 4th columns are left unchanged
            Z_join[:-1, 2:] = text_pt[:, 2:]

            top_text_node = n_text_leaves + n_text_groups
            Z_join[-1, :] = np.array(
                [0, top_text_node, 1.2, n_text_leaves + self.n_tab_cols]
            )
        return Z_join

    def shape(self, s):
        """The shape of what we return as a masker."""
        self._update_s_cache(s[self.n_tab_cols :])
        return (
            self.max_samples,
            self.n_tab_cols + sum([len(col) for col in self._tokenized_s]),
        )

    def mask_shapes(self, s):
        """The shape of the masks we expect."""
        self._update_s_cache(s[self.n_tab_cols :])
        return [(self.n_tab_cols + sum([len(col) for col in self._tokenized_s]),)]

    def feature_names(self, s):
        """The names of the features for each mask position for the given input string."""
        self._update_s_cache(s[self.n_tab_cols :])
        return [self.tab_feature_names + [v for col in self._segments_s for v in col]]


class Token:
    """A token representation used for token clustering.
    Same as Text masker but sent_no added to track which sentence the token is in"""

    def __init__(self, value, sent_no):
        self.s = value
        self.sent_no = sent_no  # added sent_no to track which sentence the token is in
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


def merge_score(group1, group2, special_tokens):
    """Compute the score of merging two token groups.

    special_tokens: tokens (such as separator tokens) that should be grouped last
    """

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


def join_dendograms(pts):
    n_leaves = [int(max(pt[:, 3])) for pt in pts]
    n_empty = sum([1 for n in n_leaves if n == 1])
    n_groups = [len(pt) for pt in pts]

    # Needed to add this because could join two together but you'd still need
    # an extra row to join the next one on. So if there are two single words
    # then we need to reduce n_empty by 1
    n_empty -= n_leaves[0] == 1 and n_leaves[1] == 1

    pt_join = np.zeros((sum(n_groups) + len(pts) - n_empty - 1, 4))
    # For the first partition tree the leaves (ie the words/features) are unchanged,
    # but the group numbers need to be shifted by the total number of leaves, less
    # the number of leaves in the first tree which are already accounted for
    pt_join[: n_groups[0], :2] = np.where(
        pts[0][:, :2] < n_leaves[0],
        pts[0][:, :2],
        pts[0][:, :2] + sum(n_leaves) - n_leaves[0],
    )
    # For the remaining trees the leaves need to be shifted by the sum of the leaves
    # of the previous trees. The group numbers need to be shifted by the total number
    # of leaves, less the number of leaves in the current tree which are already accounted for,
    # plus the number of groups in the previous trees
    g_cs = np.cumsum(n_groups)
    l_cs = np.cumsum(n_leaves)
    for pt, i in zip(pts[1:], range(1, len(pts))):
        if np.equal(pt, np.array([[0, np.inf, 0, 1]])).all():  # empty tree
            pt_join[g_cs[i - 1] : g_cs[i], :2] = np.array([l_cs[i - 1], np.inf])
        else:
            pt_join[g_cs[i - 1] : g_cs[i], :2] = np.where(
                pt[:, :2] < n_leaves[i],
                pt[:, :2] + l_cs[i - 1],
                pt[:, :2] + sum(n_leaves) - n_leaves[i] + g_cs[i - 1],
            )

    # 3rd and 4th columns are left unchanged
    pt_join[: -(len(pts) - n_empty - 1), 2:] = np.concatenate([pt[:, 2:] for pt in pts])

    # If a text feature only has one word then we need to do some extra work
    top_nodes0 = [i + sum(n_leaves) - 1 for i in g_cs]
    n_in_top_nodes0 = list(pt_join[[i + -1 for i in g_cs], -1])
    top_node_acntd_for = [False for _ in g_cs]

    # If a text feature only has one word then we join it to the top node of the
    # previous tree and increment the number of words in that top node
    for i in range(1, len(n_leaves)):
        if n_leaves[i] == 1:
            pt_join[g_cs[i] - 1, 1] = top_nodes0[i - 1]
            pt_join[g_cs[i] - 1, 2] = 1
            pt_join[g_cs[i] - 1, 3] = n_in_top_nodes0[i - 1] + 1
            top_node_acntd_for[i - 1] = True
            n_in_top_nodes0[i] += n_in_top_nodes0[i - 1]

    # Now some of the top nodes will have been accounted for, so we need to remove them
    top_nodes = [i for idx, i in enumerate(top_nodes0) if not top_node_acntd_for[idx]]
    n_in_top_nodes = [
        i for idx, i in enumerate(n_in_top_nodes0) if not top_node_acntd_for[idx]
    ]

    #########################################################################################
    # This bit is all to take into account the case where the first group is a single token
    # This is a fringe case which will only happen if the first group is a single token and
    # the tokenizer does not have a start of sentence token

    if pt_join[0, 1] == np.inf:
        # We have to check that the second group is not a single word too
        if n_leaves[1] == 1:
            # first and second groups just form a single group
            pt_join[0, 1] = 1
            pt_join[0, 3] = 2
            # pop the second group
            pt_join = np.delete(pt_join, 1, 0)
            # adjust group references: if there is a number > sum(n_leaves) then minus 1 from it
            pt_join[:, :2] = np.where(
                pt_join[:, :2] > sum(n_leaves), pt_join[:, :2] - 1, pt_join[:, :2]
            )
            # adjust top_nodes: if there is a number > sum(n_leaves) then minus 1 from it
            top_nodes = [i - 1 for i in top_nodes]
            # adjust g_cs because we use it later
            g_cs = [i - 1 for i in g_cs[1:]]
        else:
            # Connect it to the top node of the next tree
            pt_join[0, 1] = top_nodes[1]
            pt_join[0, 2] = 1
            pt_join[0, 3] = n_in_top_nodes[1] + 1
            # rearrange groups such that the first row is now behind the seconrd group
            second_grp_size = top_nodes[1] - top_nodes[0]
            new_order = (
                list(range(1, second_grp_size + 1))
                + [0]
                + list(range(1 + second_grp_size, pt_join.shape[0]))
            )
            pt_join = pt_join[new_order, :]
            # adjust group references: if there is a number >= sum(n_leaves) and <= top_nodes[1]
            # then minus 1 from it
            pt_join[:, :2] = np.where(
                (pt_join[:, :2] >= sum(n_leaves)) & (pt_join[:, :2] <= top_nodes[1]),
                pt_join[:, :2] - 1,
                pt_join[:, :2],
            )
            # redefine top_nodes and n_in_top_nodes, swapping the first two elements
            top_nodes = top_nodes[1:]
            n_in_top_nodes = n_in_top_nodes[1:]
            n_in_top_nodes[0] += 1
    #########################################################################################

    # Now we need to join the top nodes together
    joiner_rows = []
    while len(top_nodes) > 1:
        first_two = sum(n_in_top_nodes[:2])
        joiner_row = top_nodes[:2] + [0, first_two]
        joiner_rows.append(joiner_row)
        top_nodes = top_nodes[2:]
        n_in_top_nodes = n_in_top_nodes[2:]
        if len(top_nodes) > 0:
            top_nodes.insert(0, top_nodes[-1] + len(joiner_rows))
            n_in_top_nodes.insert(0, first_two)
    pt_join[g_cs[-1] :] = np.array(joiner_rows)
    pt_join[:, 2] = pt_join[:, 3]
    pt_join[:, 2] /= pt_join[:, 2].max()
    return pt_join


def partition_tree(decoded_tokens, special_tokens, sent_indices):
    """Build a heriarchial clustering of tokens that align with sentence structure.

    Note that this is fast and heuristic right now.
    TODO: Build this using a real constituency parser.
    """
    # Only difference is that we add sent_indices to the Token class
    #############################
    token_groups = [
        TokenGroup([Token(t, idx)], i)
        for i, (t, idx) in enumerate(zip(decoded_tokens, sent_indices))
    ]
    #############################
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
