import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer #, pipeline
from datasets import load_dataset, Dataset
import shap
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
import argparse
from optimum.pipelines import pipeline
from src.utils import format_text_pred, select_prepare_array_fn

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    default="imdb_genre_3",
    help="Name of model to pull from james-burton/ HuggingFace",
)
parser.add_argument('--split_for_exps', choices=['test', 'validation'], default='test')

model_name = parser.parse_args().model_name
split = parser.parse_args().split_for_exps
tab_cols = ['Year','Runtime (Minutes)', 'Rating', 'Votes', 'Revenue (Millions)','Metascore', 'Rank']
text_col = ['Description']

match model_name:
    case 'imdb_genre_0':
        cols = tab_cols + text_col
    case 'imdb_genre_1':
        cols = tab_cols
    case 'imdb_genre_6':
        cols = tab_cols
    case 'imdb_genre_5':
        cols = tab_cols
    case 'imdb_genre_7':
        cols = tab_cols + text_col
    case 'imdb_genre_2':
        cols = tab_cols + text_col
    case 'imdb_genre_3':
        cols = ['Votes', 'Revenue (Millions)', 'Metascore', 'Rank', 'Description', 'Year', 'Runtime (Minutes)', 'Rating']
    case 'imdb_genre_4':
        cols = ['Description', 'Rank', 'Metascore', 'Revenue (Millions)', 'Votes', 'Rating', 'Runtime (Minutes)', 'Year']

prepare_array_fn = select_prepare_array_fn(model_name)


class PipelineModel:
    def __init__(self, pipeline, prepare_array_fn):
        self.pipeline = pipeline
        self.prepare_array_fn = prepare_array_fn

    def predict(self, examples, load_from_cache=True):
        examples_as_strings = np.apply_along_axis(self.prepare_array_fn, 1, examples)
        preds = [
            out
            for out in self.pipeline(
                KeyDataset(Dataset.from_dict({"text": examples_as_strings}), "text"),
                batch_size=64,
            )
        ] 
        preds = np.array([format_text_pred(pred) for pred in preds])

        return preds

ds = load_dataset('james-burton/imdb_genre_prediction2')
train_df = ds['train'].to_pandas()
test_df = ds['test'].to_pandas()
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# my_model = AutoModelForSequenceClassification.from_pretrained(f'james-burton/{model_name}', num_labels=2)



X_train = train_df[cols]
X_test = test_df[cols] if split == 'test' else ds['validation'].to_pandas()[cols]
my_pipeline = pipeline(
    "text-classification", model=f'james-burton/{model_name}', tokenizer=tokenizer, device=0, accelerator="bettertransformer"
)    
my_shap_pipeline = PipelineModel(my_pipeline, prepare_array_fn)

my_explainer = shap.KernelExplainer(my_shap_pipeline.predict, X_train)
for i in range(0,len(X_test),10):
    shap_values = my_explainer.shap_values(X_test[i:i+10], seed=42)
    np.save(f"{model_name}_shap_values_{i}.npy", shap_values)
