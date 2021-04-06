# Word Mover's Distance
In this package you will find the implementation of Word Mover's Distance for a generic Word Embeddings model.

I largely reused code available in the [gensim](https://github.com/RaRe-Technologies/gensim) library, in particular the [wmdistance](https://tedboy.github.io/nlps/_modules/gensim/models/word2vec.html#Word2Vec.wmdistance) function, making it more general so that it can be used with other Word Embeddings models, such as [GloVe](https://nlp.stanford.edu/projects/glove/).

You can find a real-world usage of this package in my [news summariser repository](https://github.com/hechmik/news_summariser), where I use Word Mover's distance for finding the most similar sentences in a given news article.
# How to install

The preferred way to install this package is through [pip](https://pypi.org/project/word-mover-distance/):
```bash
pip install word-mover-distance
```
On Mac and Linux it works like a charm. On Windows, however, it is highly likely you will experience some issues: this is due to **pyemd**, which needs some C++ dependencies during build time. A quick way to solve this issue is to install "Build Tools for Visual Studio 2019" following this procedure:
- Go to the following page and download "Build Tools for Visual Studio 2019" https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019
- Double click on the .exe file once finished and select to install C++ build tools
- Check that among the suggested packages to install it is also selected "Windows 10 SDK" (the newest version is fine) as this is the critical dependency
- Once the installation has finished reopen your PowerShell/Command Prompt and retry to install the library with the original pip instruction

If storage/connectivity speed is critical for your usecase and/or you would like to know more about the issue have a look at [this Stack Overflow discussion](https://stackoverflow.com/questions/40018405/cannot-open-include-file-io-h-no-such-file-or-directory).

# Basic usage 
## Import the library:
```python
from word_mover_distance import model
```

## Initialise a Word Embedding object
You can pass the path where the model is stored:
```python
my_model = model.WordEmbedding(model_fn="/path/where/my/model/is/stored.txt")
```
or you can pass the model itself, previously loaded (assuming your model is a dictionary, whose keys are the various words and its values the vector representation of the various words):
```python
my_model = model.WordEmbedding(model=my_word_embedding_model)
```

## Compute Word Mover's distance
```python
s1 = 'Obama speaks to the media in Chicago'.lower().split()
s2 = 'The president spoke to the press in Chicago'.lower().split()
wmdistance = my_model.wmdistance(s1, s2)
1.8119693993679309
```
Remember that the ```wmdistance(s1, s2)``` method expects two ```List[str]``` as input!

