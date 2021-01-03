from typing import List
import logging
from gensim.corpora import Dictionary
import numpy as np
from pyemd import emd


def nbow(document, vocab_len, dictionary):
    d = np.zeros(vocab_len, dtype=np.double)
    nbow = dictionary.doc2bow(document)  # Word frequencies.
    doc_len = len(document)
    for idx, freq in nbow:
        d[idx] = freq / float(doc_len)  # Normalized word frequencies.
    return d


class WordEmbedding:

    def __init__(self, **kwargs):
        if "model_fn" in kwargs.keys():
            self.model = self.load_word_embedding_model(kwargs['model_fn'])
        elif "model" in kwargs.keys():
            self.model = kwargs['model']
        self.words = self.model.keys()

    def load_word_embedding_model(self, fn, encoding='utf-8'):
        """
        Return the Word Embedding model at the given path
        :param fn: path where the model of interest is stored
        :param encoding: encoding of the file of interest. Default value is utf-8
        :return:
        """
        logging.info("load_word_embedding_model >>>")
        model = {}
        with open(fn, 'r', encoding=encoding) as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                model[word] = vector
        logging.info("load_word_embedding_model <<<")
        return model

    def wmdistance(self, document1: List[str], document2: List[str]):
        """
        Compute Word Mover's distance among the two list of documents
        :param document1:
        :param document2:
        :return:
        """

        # Remove out-of-vocabulary words.
        len_pre_oov1 = len(document1)
        len_pre_oov2 = len(document2)
        document1 = [token for token in document1 if token in self.words]
        document2 = [token for token in document2 if token in self.words]
        diff1 = len_pre_oov1 - len(document1)
        diff2 = len_pre_oov2 - len(document2)
        if diff1 > 0 or diff2 > 0:
            logging.info('Removed %d and %d OOV words from document 1 and 2 (respectively).',
                         diff1, diff2)

        if len(document1) == 0 or len(document2) == 0:
            logging.info('At least one of the documents had no words that were'
                         'in the vocabulary. Aborting (returning inf).')
            return float('inf')

        dictionary = Dictionary(documents=[document1, document2])
        vocab_len = len(dictionary)

        # Sets for faster look-up.
        docset1 = set(document1)
        docset2 = set(document2)
        distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
        for i, t1 in dictionary.items():
            for j, t2 in dictionary.items():
                if not t1 in docset1 or not t2 in docset2:
                    continue
                # If the current cell is empty compute Euclidean distance between word vectors.
                if not distance_matrix[i, j]:
                    distance_matrix[i, j] = np.sqrt(np.sum((self.model[t1] - self.model[t2]) ** 2))
                    # Fill the specular cell for saving computation
                    distance_matrix[j, i] = distance_matrix[i, j]

        if np.sum(distance_matrix) == 0.0:
            # `emd` gets stuck if the distance matrix contains only zeros.
            logging.info('The distance matrix is all zeros. Aborting (returning inf).')
            return float('inf')

        # Compute nBOW representation of documents.
        d1 = nbow(document1, vocab_len, dictionary)
        d2 = nbow(document2, vocab_len, dictionary)

        wmd = emd(d1, d2, distance_matrix)
        return wmd
