# Summary

This is a project to explore what happens when you combine tabular and text data in different ways.

For example:

* Tabular data linearised to text
* Ensemble: average/ weighted average
* Combining vectorised text and tabular representations in and then running a final linear layer

How do these explanations change when I use different explanation methods?

How do these explanations change when I use different models?

When just a textual predictor is used, or just a tabular predictor is used, are the explanations similar to the explanations when both are used?

How important is the text as a whole? Swap out the whole text for another text to see how it ranks, vs swapping out individual words.

## Using the textual explanations

I will need to think of a way of linking in my textual explanations. Perhaps feeding it straight in to the textual explainer?

Noura even suggested feeding in the textual explanation as well as the text and tabular input in order to get a better explanation. Even if these experiments are not fruitful, one of the first questions that the examiners will ask is where is the link between the two projects.

## Data

* Airbnb data downloaded from <https://www.kaggle.com/datasets/tylerx/melbourne-airbnb-open-data?resource=download>
* Books data from <https://machinehack.com/hackathons/predict_the_price_of_books/data>
* Benchmark suite from <https://github.com/sxjscience/automl_multimodal_benchmark>

In [Melbourne Airbnb Price Prediction](https://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26586189.pdf) they recognise that there are multiple reviews for every listing. Reviews are concatenated together with the corresponding listing to be used as the text input. Train, val and test are split by listing id.

However, when they combine tabular and text, they only use the description, not

## Explanations

Where does explanations come in?

If the goal is to explore how explanations change when I combine the two modalities in different ways then I first need to get some explanations.

Let's look at a paper to see which explanation methods they use.
Here:

* To be clear, **saliency** is a method for visualising the explanations, but it is not an explanation method in itself. It needs to be generated using another method.
[Benchmarking and Survey of Explanation Methods for Black Box Models](https://arxiv.org/pdf/2102.13076.pdf) mention:

* LIME
* Integrated Gradients (is only valid for deep neural networks), is this the case for Layer Wise Relevance Propagation? Probably it it is talking about layers.
* Deeplift

For XGBoost they have feature importance methods built in, but SHAP says they aren't so good.

In the paper they note that LIME and integrated gradients swap a word for a space and then measure the change in the prediction.
On the [Captum website](https://captum.ai/tutorials/Bert_SQUAD_Interpret2) they look at the attention matrix for a particular layer. Tranformers have many heads and visualising the attention for each head it shows that attention matricies aren't always a good inidcator of feature importance. If I relate this back to TnT, the attention matrix is looking directly into a transformer model and the weights. The thing is that depending on the combination metric I use I won't be able to do that. I won't be able to do it unless it is simply an average, weighted or otherwise.

I could feasibly add experiments where I just use something like a bag of words, but this would be low on the priority list.

SHAP feels like it is the gold standard for which to compare things to as it is kind of guaranteed to be right, but just takes a long time. LIME will need to be run a few times. Am I just going to get to the end of this and conclude that the combination of text and tabular data just depends on the underlying model?

## How to make this as robust as I can

Well even if I say that it depends on the underlying model, the underlying model will be the same for each of the joining methods that I try. So then I can attribute the changes in the explanations to the joining method, although of course it will specific to each dataset too.

I will need to find a way to relate it across datasets. Therefore the measurements will likely be in the form of looking at the within-modality rank of the features, also looking at how much the model relies upon each modality.

To start:

* Train a model on the Airbnb tabular data :white_check_mark:
* Train a model on the Airbnb text data :white_check_mark:
* Average the two models predictions for a simple ensemble
* Get explanations for each model
  * Find out which methods do people use

## Thoughts on [DIME](https://arxiv.org/pdf/2203.02013.pdf) paper

* This actually has a lot of relevance to hierarchical
* They disentangle the LIME outputs of a multimodal model into text, image and multimodel interactions. Specifically the multimodal interactions between

## Thoughts on [Benchmarking Multimodal AutoML for Tabular Data with Text Fields](https://arxiv.org/abs/2111.02705)

* Here they have over a dozen different tabntext datasets and they compare performance between different combination methods. Namely for
  * Weighted Ensemble: weighted average of the text and tabular model predictions
  * Stack Ensemble: An additional modal is trained on the predictions of the text and tabular models

In the paper they refer to an LM that just uses the text data as Text-Net and one that uses the tabular data too (in some fashion) as Multimodal-Net.
For them All-text means that tabular features are converted into text, Fuse-early means that the tabular features are mapped by dense layers into the same embedding space as the text tokens. Fuse-late means that the text features are encoded by a tranformer into one latent space and an MLP is used to map the tabular features into another, which are then concatenated and fed into a final MLP.
Also:

* Get LM embeddings, no fine tuning on just text
* Fine tune on just text cols, then get LM embeddings
Here they even calculate feature importance for the features but it is only a small portion, and they do not go down to the word level.

Data Process methods, this comes from AutoML/AutoGluon:

* Missing values for catergorial features are represented as a new "Unknown" category
* Missing text fields are handled as an empty string
* Categorial features with more than 20 unique values are counted as a text string

## Where to go from here

* I have a few ways to combine tabular and text and I have a few potential datasets that I can try. Is the path forward to train these models first? The goal here is to compare explanations so I think I need to start cranking them out, or at least develop a pipeline for doing so. I need to also gauge how long the explanations will take
* From the benchmark dataset I have the performance from the different datasets

If we think throught the IMDB dataset

* There are 7 numerical columns and 3 text columns
* Train a model on the numerical columns
* Train a model on the text columns, maybe just use one of the columns.
* Get the SHAP explanations for several

## March 14th

I can now generate SHAP explanations for text columns as a whole when combining them in a 50/50 way. Doing the same but for a different weighting should be trivial. If I were to do a stack ensemble (to use the language from above), then I would need to make some edits. The trouble I think is just compute time if I am to run 68k examples through BERT. Currently I can do it speedily  by recognising that there is an independence in the ensemble models, so therefore I can run them seperately. In a stack ensemble I think I should be able to find some degree of separation so it should not take that long. The tricky one will be if it is all transformed into text.

Could be an interesting side bit, what is the difference between the SHAP explanations when I have tabular columns as text and use the text method (removing words) vs when I have the tabular columns as text and use the tabular method (sampling different nums).

Next move: eventually I need to bulk generate these explanations but right now maybe I just get a single explanation, or could try a set if it only takes a few minutes, for balanced, weighted and stack ensemble.

In order to do that I need to first find out which is the best weight for combining the two models. I can try different weights and see how the explanations change but it is still useful to find out the most effective weight.

## March 16th

What am I trying to achieve?
I will be able to see the different focuses of the models when they are combined in different ways.

How can I embed the text explanations?
I would have to get the explanations for the tabular model, say, then train a new model with just the explanation such as to predict again. Can I use the training set for both?

If I have a training set X_train to predict y_train, I will produce a bunch of explanations. It will be explaining the predictions of seen examples so therefore the predictions will likely be a lot more confident (overfitted) than if it were for unseen examples.

Then I have a new training set of exp(X_train) which would still be predicting y_train. It might be entirely crap at predicting y_train, but whatever. If I still test on the unseen validation and test sets then that's still okay.

As for how it fits in with this new multi-model explanation business, maybe the story can start off with that with just using a transformer all-text is bad and focuses too much on on the description. I generate explanations for X_train (and X_val and X_test) and then generate textual explanations based off of those. Then I concatenate the descriptions and the textual explanations and train a new model on that.

How can I combine with the different multi-modal explanations?
*

## March 20th

How can we combine the different modalities in order to get the overall one. I can get the overall text explanation, but how do I get the overall explanation at a word level alongside the tabular features? I could look at doing the masking thing that happens for the pure text features. So what would that look like? For text you run it a bunch of times but swapping the words for blanks and seeing how the prediction changes. For tabular we sample from the background distribtution. It would be interesting to see if there are any sort of cross modality interactions. 

For the full text transformer, how can I tell the model to focus more on the other features

Noura meeting:

* Train two transformers, one on the text and one on the tabular data, then do the explanations
* How does the order of the input effect the explanations?
* Does it make a difference to explanations if I pass the tabular features in multiple times when training the model?

## March 21st

Models:

* [x] tabular only, as text, in a transformer (imdb_genre_1)
* [x] tabular only, as text, in a transformer, with the tabular features repeated x2 and x5 (imdb_genre_6, imdb_genre_5)
* [x] all features as text, in a transformer, with the tabular features repeated x2 and x5 (imdb_genre_7, imdb_genre_2)
* [x] all features as text, in a transformer, reordered middle and reverse (imdb_genre_3, imdb_genre_4)

Explanations:

* [x] tab as text (imdb_genre_1) word style
* [x] tabx2_nodesc (imdb_genre_6) word style
* [x] tabx5_nodesc (imdb_genre_5) word style
* [x] tab as text (imdb_genre_1) tabular style
* [x] tabx2_nodesc (imdb_genre_6) tabular style
* [x] tabx5_nodesc (imdb_genre_5) tabular style
* [x] tabx2 (imdb_genre_7) tabular style
* [x] tabx5 (imdb_genre_2) tabular style
* [x] reorder1 (imdb_genre_3) tabular style
* [x] reorder2 (imdb_genre_4) tabular style
~~* [ ] all as text (imdb_genre_0) tabular style on validation set~~

In order to test the above I need to do:

* [x] np load the exps
* [x] find the data for them. I needed this data when I trained the models so look in the gen exps file
* [x] do shap summary plots

## March 22nd

I want to explore to see if performance will improve if I pass the features into the model according to how important they are to the decision (on average). I must keep the test set seperate, so I will have to use the validation set to get explanations. It doesn't make too  much sense to do it including the text description becuase the transformer model basically ignore the other features when it is present.

* [x] imdb_genre_1 tabular style on validation set
* [x] reorder and retrain based on importance values

The problem that I want to tackle is to be able to get an overall explanation which puts words alongside categorical features. This could be a technical challenge, telling the model to treat some features a certain way and some a different way. 

Two questions:

* How does lgb handle missing values?
  * [x] Search online
  ~~* [ ] Create a test case and make a prediction~~
* For simple ensembles, can we recover the feature importance values simply by multiplying together the feature importance values of the individual models?
  * [x] Look at average shap values for tab features, in lgb and how they compare when multiplied by 0.25, 0.5 and 0.75

Missing values that are unseen during training are treated as belonging to the 'other' non-split category and are always put to the right (for categorical variables). For numerical variables the missing val is converted to 0.

* Order of features in text model seems important:
  * [x] imdb_genre_0 tabular style on validation set
  * [x] reorder and retrain based on importance values
* Can I get a word level explanation that includes the text features
  * [x] Get a set of word level explanations for the text only model
  ~~* [ ] Combine them with the tabular features explanations~~

## March 23rd

I have been able to get an output for the text and tabular combination.

## March 24th

When combining the text and tabular outputs in a simple ensemble (linear transformation) the combination of explanations is trivial. The feature importance values and the expected value form a linear y = mx + c relationship, therefore when the output is transformed linearly, eg y = 0.5(text) + 0.5(tab) I can simply multiply both explanation equations together to form a new expected value and a new set of feature importance values.

I can display the output in terms of a word level explanation using the shap library, but only when I remove the `hierarchical_values`, which is the part of the explanation that groups word-pieces together. There may well be a way to set it up for the combined tabular values too but at the moment it is simpler just to remove it.

I feel like I am now in a similar place to before: I want to test the relationship between the individual model explanations and the global explanations when combining the two modalities in a stack ensemble. The current thing that we are trying to solve is how can we present an explanation that is on a word level with text and tabular data. Options:

1. Calculate them both independently and then combine them in some fashion
2. Think of a method to do both a word level and a tabular level explanation at the same time

### 1

In order to work out a method to combine them I first need to get an answer of what they are supposed to look like together. DIME calculated 3 LIME explanations, one where the image is held constant and the text is perturbed, one vice versa and one where both are perturbed. They then calculated the multimodal interaction by taking the different between the combined and the individual. We do this when the two modalities are not independent. For the case of a simple ensemble they are: perturbing the text will not effect the contribution of the tabular features and vice versa. Therefore the multimodal interaction should be 0.

How does SHAP work for tabular
 data: 
I should use the shap.kmeans function: it runs the scikit learn kmeans function to get k (go for 100) clusters of the data. The data is then rounded to the nearest datapoint in the dataset. It is then made into a DenseData object which takes the cluster centres, the group names and weights depending on how many datapoints are in each cluster.

If I had a single background reference then I would be able to do a combined text and tabular explanation, surely. I would just substitute the tabualr features for its background reference and swap the text feature for the [MASK] token.

The question is, when there is a background dataset and I am looking to 'remove' one feature, does it randomly pick from the background dataset or does it sample multiple times?

~~Text examples where the tokenisation happens within the predict function eg https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/text.html use a partition explainer. Due to this example doing the tokensation within the predict function I think that I will be able to have some success using this method. I just need to work out how to deal with the background dataset. On Monday~~

Tokenisation does happen in the predict function but the masking is still the same in the other part of the explainer, still need some thinking.

~~* [ ] Make a reference dataset of just the kmeans centroids and~~

## March 27th

How does shap explain text?

* It uses a Partition explainer. The `__call__()` function of the parent `Explainer` package is called but that just converts a Dataframe/dataset into lists and retreives the right column. It then loops over all rows which are to be explained and calls `explain_row()` from `Partition` 
* As it does it one row at a time, the effect of the removal of one or more of the words is calculated for the output of that particular row. Word masked, prediction made, other word masked, prediction made, etc. We cannot do this for tabular becaause there is no background data. 

Imagine there is a sentence to be explained and then a single tabular feature. We would calcualte the words as normal but then we also need to vary to tabular instance. Because we cannot replace it with a single mask token we have to sample it. 

## March 29th

* [x] Using the brute force kernel shap explain a single row using a single reference value
~~* [ ] Explain a single row using multiple reference values~~

Looking more closely and I am seeing that text shap values are not calculated through the permutation explainer, but instead through the partition explainer which makes groups of words and then substitutes them out as a whole when calculatin the presence or not of certain features in the shap val calculation.

I think the steps will be to first create an exact brute force explainer which does the SHAP values for everything. But this may well take ages as it will treat each word invividually. In theory I can do a partition explainer for the combined two, but in order to do so I need a way of calculating the similarity between words and the tabular features. It might be the case that I just split the clustering tree into two parts, splitting tab and text at the top level. I think this means that clusters split at the top level will treat each other as a whole, ie (i think) the tabular features will treat the words as a single feature. Same as saying "Given the text is the way that it is, how do the tabular features affect the output". 

If I am calculating them seperately (if we are talking about all as text, for example) and I am holding the text constant, then I'd be using the sampling method to get the shap values of the tabular features.

## March 30th

* [x] Get a clustering output for the text and tabular features