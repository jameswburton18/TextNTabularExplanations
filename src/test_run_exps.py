import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from datasets import load_dataset, Dataset
import shap
from transformers.pipelines.pt_utils import KeyDataset

tab_cols = ['Year','Runtime (Minutes)', 'Rating', 'Votes', 'Revenue (Millions)','Metascore', 'Rank']
text_cols = ['Description']

ds = load_dataset('james-burton/imdb_genre_prediction2')

train_df = ds['train'].to_pandas()
test_df = ds['test'].to_pandas()

all_text_model = AutoModelForSequenceClassification.from_pretrained('james-burton/imdb_genre_all_text', num_labels=2)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
  
def format_text_pred(pred):
    if pred['label'] == 'LABEL_1':
        return np.array([1-pred['score'], pred['score']])
    else:
        return np.array([pred['score'], 1-pred['score']])

X_train = train_df[tab_cols + text_cols]
X_test = test_df[tab_cols + text_cols]

cols = [
    "Year",
    "Runtime (Minutes)",
    "Rating",
    "Votes",
    "Revenue (Millions)",
    "Metascore",
    "Rank",
    "Description",
]

all_text_pipeline3 = pipeline(
    "text-classification", model=all_text_model, tokenizer=tokenizer, device=0
)


def array_to_string(array):
    return np.array(
        " | ".join([f"{col}: {val}" for col, val in zip(cols, array)]), dtype="<U512"
    )


def data(dataset):
    for i in range(len(dataset)):
        yield dataset["text"][i]


class AllTextModel:
    def __init__(self, text_to_pred_dict=None):
        self.text_to_pred_dict = text_to_pred_dict
        self.text_pred_len = 0

    def predict_both(self, examples, load_from_cache=True):
        examples_as_strings = np.apply_along_axis(array_to_string, 1, examples)
        # Check if we have already predicted these examples
        # if load_from_cache:
        #     preds = [self.text_to_pred_dict.get(desc, None) for desc in examples_as_strings]
        #     if all(pred is not None for pred in preds):
        #         return np.array(preds)

        preds = [
            out
            for out in all_text_pipeline3(
                KeyDataset(Dataset.from_dict({"text": examples_as_strings}), "text"),
                batch_size=64,
            )
        ]  # , total=len(examples))]
        preds = np.array([format_text_pred(pred) for pred in preds])

        return preds


# %%
all_text_test_model = AllTextModel()

kernel_explain_all_text = shap.KernelExplainer(all_text_test_model.predict_both, X_train)
for i in range(30,len(X_test)-10,10):
    kernel_shap_values_all_text = kernel_explain_all_text.shap_values(X_test[i:i+10], nsamples=1000, seed=42)
    np.save(f"kernel_shap_values_all_text_{i}.npy", kernel_shap_values_all_text)


# val_df = ds['validation'].to_pandas()
# X_train_tab = train_df[tab_cols]
# y_train = train_df[label_cols]
# X_test_tab = test_df[tab_cols]
# y_test = test_df[label_cols]
# X_val_tab = val_df[tab_cols]
# y_val = val_df[label_cols]

# tab_model = lgb.LGBMClassifier(random_state=42)
# tab_model.fit(X_train_tab,y_train)
# y_pred = tab_model.predict(X_test_tab)
# y_pred_probs = tab_model.predict_proba(X_test_tab)
# y_pred_val = tab_model.predict(X_val_tab)
# y_pred_probs_val = tab_model.predict_proba(X_val_tab)


# print('Accuracy: ', np.mean(y_test.values.flatten() == y_pred))
# print('ROC AUC: ', roc_auc_score(y_test, y_pred_probs[:,1]))
# print('Accuracy: ', np.mean(y_val.values.flatten() == y_pred_val))
# print('ROC AUC: ', roc_auc_score(y_val, y_pred_probs_val[:,1]))

# # %% [markdown]
# # ## Text

# # %%
# text_model = AutoModelForSequenceClassification.from_pretrained('../models/imdb_genre/frosty-night-14/checkpoint-22', num_labels=2)
# tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
# # Tokenize the dataset
# def encode(examples):
#     return {
#         "labels": np.array([examples['Genre_is_Drama']]),
#                 **tokenizer(examples['Description'], truncation=True, padding="max_length")}
# ds = ds.map(encode)
# trainer = Trainer(model=text_model)
# text_preds = F.softmax(torch.tensor(trainer.predict(ds['test']).predictions), dim=1).numpy()

# print('Accuracy: ', np.mean(np.argmax(text_preds, axis=1) == y_test.values.flatten()))
# print('ROC AUC: ', roc_auc_score(y_test.values.flatten(), text_preds[:,1]))

# # %%
# text_pipeline = pipeline('text-classification', model=text_model, tokenizer=tokenizer, device=0)

# # text_test_preds = np.array([format_text_pred(pred) for pred in 
# text_pipeline(list(ds['test']['Description']))

# # %% [markdown]
# # ## Weighted Ensemble - Even weighting

# # %%
# tab_weight = 0.5

# text_preds = trainer.predict(ds['test']).predictions
# tab_preds = tab_model.predict_proba(X_test_tab)
# ensemble_preds = tab_weight * tab_preds + (1 - tab_weight) * text_preds
# print('Accuracy: ', np.mean(np.argmax(ensemble_preds, axis=1) == y_test.values.flatten()))
# print('ROC AUC: ', roc_auc_score(y_test.values.flatten(), ensemble_preds[:,1]))

# # %% [markdown]
# # ## Weighted Ensemble - Grid search for best weight

# # %%
# # Text
# text_preds = F.softmax(torch.tensor(trainer.predict(ds['validation']).predictions), dim=1).numpy()

# # Tab
# val_df = ds['validation'].to_pandas()
# X_val = val_df[tab_cols]
# y_val = val_df[label_cols]
# tab_preds = tab_model.predict_proba(X_val)


# # %%

# roc_results = {}
# acc_results = {}
# for i in range(0,101,5):
#     tab_prop = i/100
#     text_prop = 1 - tab_prop
#     preds = tab_prop * tab_preds + text_prop * text_preds
#     roc_results[i] = roc_auc_score(y_val.values.flatten(), preds[:,1])
#     acc_results[i] = np.mean(np.argmax(preds, axis=1) == y_val.values.flatten())

# best_roc_weight = list(roc_results.keys())[list(roc_results.values()).index(max(roc_results.values()))]/100
# plt.plot(list(roc_results.keys()), list(roc_results.values()))
# plt.show()
# # plt.plot(list(acc_results.keys()), list(acc_results.values()))
# # plt.show()
# print('Best ROC AUC: ', max(roc_results.values()), ' at ', best_roc_weight)

# # %%
# text_preds = trainer.predict(ds['test']).predictions
# tab_preds = tab_model.predict_proba(X_test_tab)
# ensemble_preds = best_roc_weight * tab_preds + (1 - best_roc_weight) * text_preds
# print('Accuracy: ', np.mean(np.argmax(ensemble_preds, axis=1) == y_test.values.flatten()))
# print('ROC AUC: ', roc_auc_score(y_test.values.flatten(), ensemble_preds[:,1]))

# # %% [markdown]
# # ## Training a stack ensemble

# # %% [markdown]
# # [Good link on stack ensemble](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html)

# # %%
# text_pipeline = pipeline('text-classification', model=text_model, tokenizer=tokenizer, device=0)

# # Training set is the preditions from the tabular and text models
# text_val_preds = F.softmax(torch.tensor(trainer.predict(ds['validation']).predictions), dim=1).numpy()
# tab_val_preds = tab_model.predict_proba(X_val_tab)
# text_test_preds = F.softmax(torch.tensor(trainer.predict(ds['test']).predictions), dim=1).numpy()
# tab_test_preds = tab_model.predict_proba(X_test_tab)
# text_train_preds = F.softmax(torch.tensor(trainer.predict(ds['train']).predictions), dim=1).numpy()
# tab_train_preds = tab_model.predict_proba(X_train_tab)


# # %%
# # add text and tabular predictions to the val_df
# stack_val_df = val_df[tab_cols]
# stack_val_df['tab_preds'] = tab_val_preds[:,1]
# stack_val_df['text_preds'] = text_val_preds[:,1]
# stack_test_df = test_df[tab_cols]
# stack_test_df['tab_preds'] = tab_test_preds[:,1]
# stack_test_df['text_preds'] = text_test_preds[:,1]
# # stack_train_df = train_df[tab_cols]
# # stack_train_df['tab_preds'] = tab_train_preds[:,1]
# # stack_train_df['text_preds'] = text_train_preds[:,1]

# stack_model = lgb.LGBMClassifier(random_state=42)
# stack_model.fit(stack_val_df, y_val)
# # stack_model.fit(stack_train_df, y_train)

# stack_pred = stack_model.predict_proba(stack_test_df)
# print('Accuracy: ', np.mean(y_test.values.flatten() == np.argmax(stack_pred, axis=1)))
# print('ROC AUC: ', roc_auc_score(y_test, stack_pred[:,1]))

# stack_pred_val = stack_model.predict_proba(stack_val_df)
# print('Accuracy: ', np.mean(y_val.values.flatten() == np.argmax(stack_pred_val, axis=1)))
# print('ROC AUC: ', roc_auc_score(y_val, stack_pred_val[:,1]))

# y_pred = tab_model.predict(X_test_tab)
# y_pred_probs = tab_model.predict_proba(X_test_tab)
# print('Accuracy: ', np.mean(y_test.values.flatten() == y_pred))
# print('ROC AUC: ', roc_auc_score(y_test, y_pred_probs[:,1]))

# %% [markdown]
# ## All as text

# %%

  
# # Tokenize the dataset
# def encode(examples):
#     return {
#         "labels": np.array([examples['Genre_is_Drama']]),
#                 **tokenizer(examples['text'], truncation=True, padding="max_length")}
# all_text_ds = load_dataset('james-burton/imdb_genre_prediction_all_text')
# all_text_ds = all_text_ds.map(encode)
# all_text_trainer = Trainer(model=all_text_model)
# all_text_preds = F.softmax(torch.tensor(all_text_trainer.predict(all_text_ds['test']).predictions), dim=1).numpy()

# print('Accuracy: ', np.mean(np.argmax(all_text_preds, axis=1) == y_test.values.flatten()))
# print('ROC AUC: ', roc_auc_score(y_test.values.flatten(), all_text_preds[:,1]))

# %% [markdown]
# # SHAP

# %% [markdown]
# ## Tab

# %%
# tab_explainer = shap.TreeExplainer(tab_model)
# tab_shap_values = tab_explainer.shap_values(X_test_tab)
# shap.summary_plot(tab_shap_values, X_test_tab, plot_type="bar")

# # %%
# # # Select the shap values for the sixth instance
# instance_idx = 6

# shap.waterfall_plot(shap.Explanation(values=tab_shap_values[1][instance_idx], base_values=tab_explainer.expected_value[1], data=X_test_tab.iloc[instance_idx]))

# %% [markdown]
# ## Text

# %%
# ds_test_text = ds['test'].rename_columns({'Description': 'text', 'Genre_is_Drama': 'label'})
#     # keep only the text and label columns
# ds_test_text = ds_test_text.select_columns(['text', 'label'])


# # %%
# text_model = AutoModelForSequenceClassification.from_pretrained(
#     "../models/imdb_genre/frosty-night-14/checkpoint-22", num_labels=2
# )
# # text_model = torch.compile(text_model)
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# text_pipeline = pipeline('text-classification', model=text_model, tokenizer=tokenizer, device="cuda:0")


# # define a prediction function
# def f(x):
#     tv = torch.tensor(
#         [
#             tokenizer.encode(v, padding="max_length", max_length=500, truncation=True)
#             for v in x
#         ]
#     ).cuda()
#     outputs = text_model(tv)[0].detach().cpu().numpy()
#     scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
#     return scores

# def text_pred_fn(examples):
#     dataset = Dataset.from_dict({'text': examples})
#     # put the dataset on the GPU
    
#     preds = [out for out in text_pipeline(KeyDataset(dataset, "text"), batch_size=64)]
#     preds = np.array([format_text_pred(pred) for pred in preds])
#     return preds

# text_explainer = shap.Explainer(text_pred_fn, tokenizer)
# # text_explainer = shap.Explainer(f, tokenizer)
# text_shap_values = text_explainer(ds_test_text[:], fixed_context=1, batch_size=64)

# # %%
# shap.plots.text(text_shap_values[0])
# shap.plots.text(text_shap_values[134])
# shap.plots.text(text_shap_values[1])

# %% [markdown]
# If we want to get an overall picture of how important the text is to the decision then how does it work if we treat the text like a single block?
# 
# If it is just a simple weighted ensemble then it will not be too useful as the the masked/hidden text, which in shap I'm pretty sure is just the word (or in our case entire text) removed, will produce the same output from the transformer part every time. Actually we can't do it like that as replacing the word with a blank makes sense but replacing the entire text feels strange. How does it work for tabular features? 
# 
# [Kernel shap consists of five steps](https://christophm.github.io/interpretable-ml-book/shap.html#kernelshap):
# * Sample coalitions z′k ∈ {0,1}M, k ∈ {1, ..., K} where 1 represents the presence of a feature in the coalition and 0 represents the absence.
# * Obtain the model prediction for each z′k by converting it to the original feature space and applying the model ^f: ^f(hx(z′k)).
# * Compute the SHAP kernel weight for each z′k.
# * Fit a weighted linear model using the computed weights.
# * Return the Shapley values ϕk, which are the coefficients obtained from the linear model.

# %% [markdown]
# If the text column was just another tabular column... well actually if we treat it like a categorical column then we could say what would the output be with it removed ie completely blank. 
# 
# For example if blank makes the text model predict 0.2
# example makes text model predict 0.8
# and it was a 50/50 split in the training data then the shap value for text would be 0.5 * (0.8 - 0.2) = 0.3

# %% [markdown]
# For continuous variables SHAP uses a weighted average of other observations in a background dataset. The weights are determined by a kernel function that measures how similar an observation is to the original one.
# 
# Still need to think about how it all fits in, but I can pre-calculate the similarity to other data points before hand using a kernel function such as cosine similarity. (A kernel function is simply a  function is used to measure the similarity between two vectors.) I can pre-calculate the similarity but I would still need to calculate the model prediction for each z′k by converting it to the original feature space and applying the model ^f: ^f(hx(z′k)).
# 
# It should be a linear transformation between the explanations of the tab model and the explanations of the whole model, probably the weight of each of the branches multiplied together.

# %% [markdown]
# For when it is not a weighted combination of the two modalities:
# * Should I be able to take advantage of TreeSHAP to get the shap values for the tabular part? Not sure. Worrying about making things faster is probably premature optimisation.
# * This kind of relates back to the old hierarchical stuff, I can look at just the top model and see how much it relies on the text input vs the tabular

# %% [markdown]
# What is the next steps?
# * I want to see how the explanations change when I combine them in different ways, but in order to do so I do need to buld an implementation of SHAP where it does something like I am talking about above
# * I can't just do what I was doing with the hierarchical stuff because text is not the same as a categorical variable, not for transformers anyway

# %% [markdown]
# Steps:
# * Changes dataframe into array
# * Goes over each sample in the dataset you are explaining
# * In the explain function the meat seems to come if more than one feature varies, otherwise shortcuts happen to make it faster
# * 
# 
# In self.run() a new dataframe is created from the new samples and then this is passed through the model to create new predictions. If I have my data here in whatever format, I think I can make the prediction in the prediction function. Therefore it is before this when the sampling and the weighting is done where I will need to make changes.

# %% [markdown]
# If the text was just another column, then I would 

# %% [markdown]
# ## Joint

# %%
# text_model = AutoModelForSequenceClassification.from_pretrained('models/imdb_genre/frosty-night-14/checkpoint-22', num_labels=2)
# tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
# # Tokenize the dataset
# def encode(examples):
#     return {
#         "labels": np.array([examples['Genre_is_Drama']]),
#                 **tokenizer(examples['Description'], truncation=True, padding="max_length")}
# ds = ds.map(encode)



# %%


# class Model():
#     def __init__(self, text_to_pred_dict=None):
#         self.text_to_pred_dict = text_to_pred_dict
#         self.text_pred_len = 0
        
#     def predict_both(self, examples, text_weight=0.5, load_from_cache=True):
#         tab_examples = examples[:,:-1]
#         tab_preds = tab_model.predict_proba(tab_examples)
#         text_examples = examples[:,-1]
        
#         desc_dict = {}
#         for i, desc in tqdm(enumerate(text_examples)):
#             if desc not in desc_dict:
#                 desc_dict[desc] = [i]
#             else:
#                 desc_dict[desc].append(i)
        
#         if load_from_cache:
#             text_preds = np.array([self.text_to_pred_dict[desc] for desc in desc_dict.keys()])    
            
#         else:
#             text_preds = text_pipeline(list(desc_dict.keys()))
#             text_preds = np.array([format_text_pred(pred) for pred in text_preds])
                            
        
#         expanded_text_preds = np.zeros((len(text_examples), 2))
#         for i, (desc, idxs) in enumerate(desc_dict.items()):
#             expanded_text_preds[idxs] = text_preds[i]
        
#         # Combine the predictions, multiplying the text and predictions by 0.5
#         preds = text_weight * expanded_text_preds + (1-text_weight) * tab_preds
#         return preds

# class StackModel():
#     def __init__(self, text_to_pred_dict=None):
#         self.text_to_pred_dict = text_to_pred_dict
#         self.text_pred_len = 0
        
#     def predict_both(self, examples, load_from_cache=True):
#         tab_examples = examples[:,:-1]
#         tab_preds = tab_model.predict_proba(tab_examples)
#         text_examples = examples[:,-1]
        
#         desc_dict = {}
#         for i, desc in tqdm(enumerate(text_examples)):
#             if desc not in desc_dict:
#                 desc_dict[desc] = [i]
#             else:
#                 desc_dict[desc].append(i)
        
#         if load_from_cache:
#             text_preds = np.array([self.text_to_pred_dict[desc] for desc in desc_dict.keys()])    
            
#         else:
#             text_preds = text_pipeline(list(desc_dict.keys()))
#             text_preds = np.array([format_text_pred(pred) for pred in text_preds])
                            
        
#         expanded_text_preds = np.zeros((len(text_examples), 2))
#         for i, (desc, idxs) in enumerate(desc_dict.items()):
#             expanded_text_preds[idxs] = text_preds[i]
        
#         # Stack
#         stack_examples = np.hstack([tab_examples, tab_preds[:,1:], expanded_text_preds[:,1:]])
#         stack_preds = stack_model.predict_proba(stack_examples)
        
#         return stack_preds
    


# # %%

# # %% [markdown]
# # As the predictions from the transformer model will always be the same for a given input, I precalculate the predictions for each of the samples in the background dataset (X_train) and the prediction dataset (X_test). This means that an explanation run through takes 100s instead of 10 mins.

# # %%
# X_test_train = pd.concat([X_train, X_test])
# text_preds = text_pipeline(list(X_test_train['Description']))


# text_preds = np.array([format_text_pred(pred) for pred in text_preds])
            
# text_to_pred_dict = {desc: preds for desc, preds in zip(list(X_test_train['Description']), text_preds)}

# # %% [markdown]
# # ### Even weighting

# # %%
# test_model = Model(text_to_pred_dict)

# kernel_explain = shap.KernelExplainer(lambda x: test_model.predict_both(x,text_weight=0.5), X_train)
# kernel_shap_values = kernel_explain.shap_values(X_test) #, nsamples=100)
# shap.summary_plot(kernel_shap_values, X_test, plot_type="bar")

# # %% [markdown]
# # ### Text weight: .25

# # %%
# test_model = Model(text_to_pred_dict)

# kernel_explain_25 = shap.KernelExplainer(lambda x: test_model.predict_both(x,text_weight=0.25), X_train)
# kernel_shap_values_25 = kernel_explain_25.shap_values(X_test)
# shap.summary_plot(kernel_shap_values_25, X_test, plot_type="bar")

# # %% [markdown]
# # ### Text weight: .75

# # %%
# test_model = Model(text_to_pred_dict)

# kernel_explain_75 = shap.KernelExplainer(lambda x: test_model.predict_both(x,text_weight=0.75), X_train)
# kernel_shap_values_75 = kernel_explain_75.shap_values(X_test)
# shap.summary_plot(kernel_shap_values_75, X_test, plot_type="bar")

# # %% [markdown]
# # ## Stack Ensemble

# # %%
# kernel_shap_values_stack

# # %%
# stack_test_model = StackModel(text_to_pred_dict)

# kernel_explain_stack = shap.KernelExplainer(stack_test_model.predict_both, X_train)
# kernel_shap_values_stack = kernel_explain_stack.shap_values(X_test)
# shap.summary_plot(kernel_shap_values_stack, X_test, plot_type="bar")

# # %%
# len(text_to_pred_dict)

# # %%
# instance_idx = 0
# instance_shap_values = kernel_shap_values[1][instance_idx]

# # Get the base value for the model
# base_value = kernel_explain.expected_value[1]

# # Generate a force plot for the sixth instance
# shap.force_plot(base_value, instance_shap_values, X_test.iloc[instance_idx])

# # %%


# # %%
# shap.waterfall_plot(
#     shap.Explanation(
#         values=kernel_shap_values[1][0],
#         base_values=kernel_explain.expected_value[1],
#         data=X_test.iloc[0],
#     )
# )


# # %% [markdown]
# # There are only 681 unique values in the text column, but I am required to run the the model on 68,000 samples 

# # %% [markdown]
# # ## All as text

# # %%
# all_text_pipeline = pipeline('text-classification', model=all_text_model, tokenizer=tokenizer, device=0)
# all_text_explainer = shap.Explainer(all_text_pipeline)
# all_text_shap_values = all_text_explainer(list(all_text_ds['test']['text'])[:15])

# # %%
# all_text_pipeline = pipeline('text-classification', model=all_text_model, tokenizer=tokenizer, device=0)
# all_text_explainer = shap.Explainer(all_text_pipeline, fixed_context=1)
# all_text_shap_values = all_text_explainer(list(all_text_ds['test']['text'])[:15])

# # %%
# all_text_pipeline = pipeline('text-classification', model=all_text_model, tokenizer=tokenizer, device='cuda:0')
# all_text_explainer = shap.Explainer(all_text_pipeline)  #, fixed_context=True)
# all_text_shap_values = all_text_explainer(list(all_text_ds['test']['text'])[:15])

# # %%
# shap.plots.text(all_text_shap_values[0,:,"LABEL_1"])

# # %%


# # %%