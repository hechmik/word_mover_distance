# Word Mover's Distance
In this package you will find the implementation of Word Mover's Distance for a generic Word Embeddings model.

I largely reused code available in the [gensim](https://github.com/RaRe-Technologies/gensim) library, in particular the [wmdistance](https://tedboy.github.io/nlps/_modules/gensim/models/word2vec.html#Word2Vec.wmdistance) function, making it more general so that it can be used with other Word Embeddings models, such as [GloVe](https://nlp.stanford.edu/projects/glove/).

You can find a real-world usage of this package in my [news summariser repository](https://github.com/hechmik/news_summariser), where I use Word Mover's distance for finding the most similar sentences in a given news article.
# How to install

The preferred way to install this package is through pip:
```bash
pip install word-mover-distance
```

# Basic usage 
Import the library:
```python
import word_embedding.model as model
```

## Initialise a Word Embedding object
You can pass the path where the model is stored:
```python
model = model.WordEmbedding(model_fn="/path/where/my/model/is/stored.txt")
```
or you can pass the model itself, previously loaded (assuming your model is a dictionary, whose keys are the various words and its values the vector representation of the various words):
```python
model = model.WordEmbedding(model=my_word_embedding_model)
```

## Compute Word Mover's distance
```python
s1 = 'Obama speaks to the media in Chicago'.lower().split()
s2 = 'The president spoke to the press in Chicago'.lower().split()
wmdistance = model.wmdistance(s1, s2)
1.8119693993679309
```
Remember that the ```wmdistance(s1, s2)``` method expects two ```List[str]``` as input!

