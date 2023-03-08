# Summary
This is a project to explore what happens when you combine tabular and text data in different ways.

For example:
* Tabular data linearised to text
* Ensemble: average/ weighted average
* Combining vectorised text and tabular representations in and then running a final linear layer

How do these explanations change when I use different explanation methods?

How do these explanations change when I use different models?

When just a textual predictor is used, or just a tabular predictor is used, are the explanations similar to the explanations when both are used?

## Using the textual explanations
I will need to think of a way of linking in my textual explanations. Perhaps feeding it straight in to the textual explainer?

Noura even suggested feeding in the textual explanation as well as the text and tabular input in order to get a better explanation. Even if these experiments are not fruitful, one of the first questions that the examiners will ask is where is the link between the two projects.

## Data
* Airbnb data downloaded from https://www.kaggle.com/datasets/tylerx/melbourne-airbnb-open-data?resource=download
* Books data from https://machinehack.com/hackathons/predict_the_price_of_books/data

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
- Train a model on the Airbnb tabular data :white_check_mark:
- Train a model on the Airbnb text data :white_check_mark:
- Average the two models predictions for a simple ensemble
- Get explanations for each model
   - Find out which methods do people use 

## Thoughts on [DIME](https://arxiv.org/pdf/2203.02013.pdf) paper
* This actually has a lot of relevance to hierarchical
* They disentangle the LIME outputs of a multimodal model into text, image and multimodel interactions. Specifically the multimodal interactions between 