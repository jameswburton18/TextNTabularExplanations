# %%
from auto_mm_bench.datasets import dataset_registry
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, Dataset

print(dataset_registry.list_keys()) 

# %%
dataset_name = 'news_popularity2'

train_dataset = dataset_registry.create(dataset_name, 'train')
test_dataset = dataset_registry.create(dataset_name, 'test')
print(train_dataset.data.head())

label_cols = train_dataset.label_columns
tab_cols = [' n_tokens_content', ' average_token_length', ' num_keywords']
text_cols = ['article_title']

X_train = train_dataset.data[tab_cols]
y_train = train_dataset.data[label_cols]
X_test = test_dataset.data[tab_cols]
y_test = test_dataset.data[label_cols]

# model = xgb.XGBRegressor()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print('RMSE: ', np.sqrt(mean_squared_error(y_test, y_pred)))
# print('R^2: ',r2_score(y_test, y_pred))

# Text only
# join text columns and label columns
train_txt = train_dataset.data[text_cols + label_cols]


# %%
train_txt = train_dataset.data[text_cols + label_cols] 
test_txt = test_dataset.data[text_cols + label_cols]

# load dataset from dataframe
train_ds = Dataset.from_pandas(train_txt)
test_ds = Dataset.from_pandas(test_txt)
train_ds = train_ds.train_test_split(test_size=0.15)
ds = DatasetDict({'train': train_ds['train'], 'validation': train_ds['test'], 'test': test_ds})
ds = ds.map(lambda x: {'label': x[label_cols[0]], 'text': x[text_cols[0]]})

model_base = 'distilbert-base-uncased'

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(
        model_base, num_labels=1, problem_type="regression"
    )
tokenizer = AutoTokenizer.from_pretrained(model_base)

# Tokenize datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

ds = ds.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='test',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    report_to="none",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    torch_compile=True, # Needs to be true if PyTorch 2.0
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
)

trainer.train()
