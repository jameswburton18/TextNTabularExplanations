# %%
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

import lightgbm as lgb
from xgboost import XGBClassifier

# %%
test_df = load_dataset('james-burton/imdb_genre_prediction2', split='test')
tab_cols = ['Year','Runtime (Minutes)', 'Rating', 'Votes', 'Revenue (Millions)','Metascore', 'Rank']
text_col = ['Description']

test_df_text = prepare_text(test_df, 'text_col_only')
test_df_tab = test_df.to_pandas()[tab_cols]

train_df = load_dataset('james-burton/imdb_genre_prediction2', split='train').to_pandas()
train_df_tab = train_df[tab_cols]
y_train = train_df['Genre_is_Drama']


# %% [markdown]
# ## Text preds

# %%
text_model = AutoModelForSequenceClassification.from_pretrained(
    "james-burton/imdb_genre_9", num_labels=2
)
# text_model = torch.compile(text_model)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
text_pipeline = pipeline('text-classification', model=text_model, tokenizer=tokenizer, device="cuda:0")

def text_pred_fn(examples):
    dataset = Dataset.from_dict({'text': examples})
    # put the dataset on the GPU
    
    preds = [out for out in text_pipeline(KeyDataset(dataset, "text"), batch_size=64)]
    preds = np.array([format_text_pred(pred) for pred in preds])
    return preds

text_explainer = shap.Explainer(text_pred_fn, tokenizer)
# text_explainer = shap.Explainer(f, tokenizer)
text_shap_values = text_explainer(test_df_text[:1], fixed_context=1, batch_size=64)
# text base values are different for each prediction

# %%
np.concatenate([text_pred_fn(np.expand_dims(x,0)) for x in test_df_text[:]]).mean(axis=0)

# %% [markdown]
# ## Tab preds

# %%
tab_model = lgb.LGBMClassifier(random_state=42)
# tab_model = XGBClassifier(random_state=42)
tab_model.fit(train_df_tab,y_train)

def tab_pred_fn(examples):
    preds = tab_model.predict_proba(examples)
    return preds

tab_explainer = shap.KernelExplainer(tab_pred_fn, train_df_tab)
tab_shap_values = tab_explainer.shap_values(test_df_tab[:1])
# tab_explainer_tree = shap.TreeExplainer(tab_model)
# tab_shap_values_tree = tab_explainer_tree.shap_values(test_df_tab)

# %% [markdown]
# We have to use kernel explainer exps bc the tree explainer uses log odds, which I don't think can be converted to probablities.
# 
# Also I can confirm that the expected values are the mean values across the entire dataset

# %%
tab_explainer.expected_value

# %% [markdown]
# ## Ensemble

# %%
class Model():
    def __init__(self, text_to_pred_dict=None):
        self.text_to_pred_dict = text_to_pred_dict
        self.text_pred_len = 0
        
    def predict_both(self, examples, text_weight=0.5, load_from_cache=True):
        tab_examples = examples[:,:-1]
        tab_preds = tab_model.predict_proba(tab_examples)
        text_examples = examples[:,-1]
        
        desc_dict = {}
        for i, desc in tqdm(enumerate(text_examples)):
            if desc not in desc_dict:
                desc_dict[desc] = [i]
            else:
                desc_dict[desc].append(i)
        
        if load_from_cache:
            text_preds = np.array([self.text_to_pred_dict[desc] for desc in desc_dict.keys()])    
            
        else:
            text_preds = text_pipeline(list(desc_dict.keys()))
            text_preds = np.array([format_text_pred(pred) for pred in text_preds])
                            
        
        expanded_text_preds = np.zeros((len(text_examples), 2))
        for i, (desc, idxs) in enumerate(desc_dict.items()):
            expanded_text_preds[idxs] = text_preds[i]
        
        # Combine the predictions, multiplying the text and predictions by 0.5
        preds = text_weight * expanded_text_preds + (1-text_weight) * tab_preds
        return preds



# %% [markdown]
# Making things quicker by pre-running the preds

# %%
X_test_train = pd.concat([train_df[tab_cols + text_col], test_df.to_pandas()[tab_cols + text_col]])
text_preds = text_pipeline(list(X_test_train['Description']))


text_preds = np.array([format_text_pred(pred) for pred in text_preds])
            
text_to_pred_dict = {desc: preds for desc, preds in zip(list(X_test_train['Description']), text_preds)}

# %%
# ensemble_model = Model(text_to_pred_dict)

# ensemble_explainer = shap.KernelExplainer(lambda x: ensemble_model.predict_both(x,text_weight=0.5), train_df[tab_cols + text_col])

# # %%
# np.concatenate([ensemble_model.predict_both(np.expand_dims(x,0),text_weight=0.5) for x in train_df[tab_cols + text_col].values]).mean(axis=0)

# # %%
# ensemble_model.predict_both(x,text_weight=0.5)

# # %%
# np.moveaxis(np.array(tab_shap_values), [0,1,2], [2,0,1])[0]

# # %%
# tab_explainer.expected_value

# # %%
# test_df_tab.iloc[0]

# np.array([f'{col}: {val}' for col, val in zip(tab_cols, test_df_tab.iloc[0])])

# # %%
# np.concatenate([new_shap_val0.values, np.moveaxis(np.array(tab_shap_values), [0,1,2], [2,0,1])[0]]).shape

# %%
new_shap_val0 = text_shap_values[0]

# %%
new_shap_val0.values = np.concatenate([0.5*new_shap_val0.values, 0.5*np.moveaxis(np.array(tab_shap_values), [0,1,2], [2,0,1])[0]])

# %%
new_shap_val0.base_values = 0.5*new_shap_val0.base_values + 0.5*tab_explainer.expected_value

# %%
new_shap_val0.data = np.concatenate([new_shap_val0.data, np.array([f'{col}: {val}' for col, val in zip(tab_cols, test_df_tab.iloc[0])])])

# %%
shap.plots.text(text_shap_values[0])

# %%

shap.plots.text(new_shap_val0)


# %%



