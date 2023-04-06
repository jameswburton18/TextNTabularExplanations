import numpy as np
import shap
from shap.maskers import Masker
from shap.maskers._text import partition_tree, Text, SimpleTokenizer
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
from .models import Model

# import faulthandler

# faulthandler.enable()


class JointMasker(Masker):
    def __init__(
        self,
        tab_df,
        max_samples=20,
        tokenizer=None,
        mask_token=None,
        collapse_mask_token="auto",
        output_type="string",
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

        # Tab
        self.output_dataframe = False
        if safe_isinstance(tab_df, "pandas.core.frame.DataFrame"):
            self.tab_feature_names = list(tab_df.columns)
            tab_df = tab_df.values
            self.output_dataframe = True

        # Tab clustering
        self.n_tab_cols = tab_df.shape[-1]
        self.tab_pt = sp.cluster.hierarchy.complete(
            sp.spatial.distance.pdist(tab_df.T, metric="correlation")
        )
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
        However, if only the text is being masked, then we do not need to sample
        from tab_df and can just take the first row
        """
        masked_tab = self.tab_mask_call(mask[: self.n_tab_cols], x[: self.n_tab_cols])
        # masked_tab = pd.DataFrame(self._masked_data.copy())
        masked_text = self.text_mask_call(mask[self.n_tab_cols :], x[self.n_tab_cols])

        masked_tab["text"] = masked_text[0][0]  #'hello' #

        return masked_tab.values

        # if mask[:self.n_tab_cols].all():
        #     self.mask
        #     return masked_tab[:1].values
        # else:
        #     return masked_tab.values

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
        mask = self._standardize_mask(mask, s)
        self._update_s_cache(s)

        # if we have a fixed prefix or suffix then we need to grow the mask to account for that
        if self.keep_prefix > 0 or self.keep_suffix > 0:
            mask = mask.copy()
            mask[: self.keep_prefix] = True
            mask[-self.keep_suffix :] = True

        if self.output_type == "string":
            # if self.mask_token == "":
            #     out = self._segments_s[mask]
            # else:
            #     #out = np.array([self._segments_s[i] if mask[i] else self.mask_token for i in range(len(mask))])
            out_parts = []
            is_previous_appended_token_mask_token = False
            sep_token = getattr_silent(self.tokenizer, "sep_token")
            for i, v in enumerate(mask):
                # mask ignores separator tokens and keeps them unmasked
                if v or sep_token == self._segments_s[i]:
                    out_parts.append(self._segments_s[i])
                    is_previous_appended_token_mask_token = False
                else:
                    if not self.collapse_mask_token or (
                        self.collapse_mask_token
                        and not is_previous_appended_token_mask_token
                    ):
                        out_parts.append(" " + self.mask_token)
                        is_previous_appended_token_mask_token = True
            out = "".join(out_parts)

            # tokenizers which treat spaces like parts of the tokens and dont replace the special token while decoding need further postprocessing
            # by replacing whitespace encoded as '_' for sentencepiece tokenizer or 'Ġ' for sentencepiece like encoding (GPT2TokenizerFast)
            # with ' '
            if safe_isinstance(self.tokenizer, SENTENCEPIECE_TOKENIZERS):
                out = out.replace("▁", " ")

            # replace sequence of spaces with a single space and strip beginning and end spaces
            out = re.sub(
                r"[\s]+", " ", out
            ).strip()  # TODOmaybe: should do strip?? (originally because of fast vs. slow tokenizer differences)

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
                # print("mask len", len(out))
                # # crop the output if needed
                # if self.max_length is not None and len(out) > self.max_length:
                #     new_out = np.zeros(self.max_length)
                #     new_out[:] = out[:self.max_length]
                #     new_out[-self.keep_suffix:] = out[-self.keep_suffix:]
                #     out = new_out

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

    def clustering(self, s=[7.7, 398972.0, "offbeat romantic comedy"]):
        text = s[-1]

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

        text_pt = partition_tree(tokens, special_tokens)
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

        # 3rd and 4th columns are left unchanged
        Z_join[: self.n_tab_groups, 2:] = self.tab_pt[:, 2:]
        Z_join[self.n_tab_groups : -1, 2:] = text_pt[:, 2:]

        # Create top join, joining the text and tab dendrograms together
        top_tab_node = self.n_tab_cols + n_text_leaves + self.n_tab_groups - 1
        top_text_node = top_tab_node + n_text_groups
        # Set similarity of top node to 1.5
        Z_join[-1, :] = np.array(
            [top_tab_node, top_text_node, 1.5, n_text_leaves + self.n_tab_cols]
        )

        return Z_join

    def shape(self, s):
        """The shape of what we return as a masker.

        Note we only return a single sample, so there is no expectation averaging.
        """
        self._update_s_cache(s[-1])
        return (self.max_samples, self.n_tab_cols + len(self._tokenized_s))

    def mask_shapes(self, s):
        """The shape of the masks we expect."""
        self._update_s_cache(s[-1])
        return [(self.n_tab_cols + len(self._tokenized_s),)]

    def feature_names(self, s):
        """The names of the features for each mask position for the given input string."""
        self._update_s_cache(s[-1])
        return [self.tab_feature_names + [v.strip() for v in self._segments_s]]


def run_shap_vals(type="text"):
    train_df = load_dataset(
        "james-burton/imdb_genre_prediction2", split="train"
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
        "Revenue (Millions)",
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

    def text_pred_fn(examples):
        dataset = Dataset.from_dict({"text": examples})
        # put the dataset on the GPU

        preds = [
            out for out in text_pipeline(KeyDataset(dataset, "text"), batch_size=64)
        ]
        preds = np.array([format_text_pred(pred) for pred in preds])
        return preds

    test_model = Model(tab_model=tab_model, text_pipeline=text_pipeline)

    # We want to explain a single row
    np.random.seed(1)
    # x = np.array([['2009.0', '95.0', '7.7', '398972.0', '32.39', '76.0', '508.0',
    #     "An offbeat romantic comedy about a woman who doesn't believe true love exists, and the young man who falls for her."]])
    x = [7.7, 398972.0, 32.39, "offbeat romantic comedy"]

    if type == "joint":
        masker = JointMasker(tab_df=train_df[tab_cols], tokenizer=tokenizer)
        pt = masker.clustering(x)

        explainer = shap.explainers.Partition(
            model=test_model.predict_both, masker=masker, partition_tree=pt
        )
        shap_vals = explainer(np.array([x], dtype=object))
        return shap_vals
    elif type == "text":
        masker = Text(tokenizer=tokenizer)
        explainer = shap.Explainer(model=text_pred_fn, masker=masker)
        shap_vals = explainer(["offbeat romantic comedy"])
        return shap_vals
    else:
        masker = Tabular(train_df[tab_cols], clustering="correlation")
        explainer = shap.explainers.Partition(model=tab_pred_fn, masker=masker)
        shap_vals = explainer(np.array([x[:-1]]))
        return shap_vals


if __name__ == "__main__":
    # run_shap_vals('tab')
    run_shap_vals("joint")
    # run_shap_vals('text')
