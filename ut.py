import unittest
import numpy as np
import word_embedding


def load_model(fn):
    model = {}
    with open(fn, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            model[word] = vector
    return model


class GloVeUT(unittest.TestCase):
    fn = "../glove.6B/glove.6B.50d.txt"

    def test_init(self):
        we_fn = word_embedding.WordEmbedding(model_fn=self.fn)
        model_fn = we_fn.model

        model = load_model(self.fn)
        we_mod = word_embedding.WordEmbedding(model=model)
        model_mod = we_mod.model

        self.assertTrue(np.array_equal(model_fn['the'],
                                       model_mod['the']),
                        "Both initialisation work and return the same vector")

    def test_wmd(self):
        we = word_embedding.WordEmbedding(model_fn=self.fn)
        sentence_obama = 'Obama speaks to the media in Chicago'.lower().split()
        sentence_president = 'The president spoke to the press in Chicago'.lower().split()
        sentence_not_related = 'Today is a nice day! What do you think?'.lower().split()
        score_similar = we.wmdistance(sentence_obama, sentence_president)
        score_not_related = we.wmdistance(sentence_obama, sentence_not_related)
        self.assertTrue(score_similar > 0, "Distance computed correctly")
        self.assertTrue(score_not_related > 0, "Distance computed correctly")
        self.assertTrue(score_not_related > score_similar,
                        "Two similar sentences have a lower score than different ones")


if __name__ == '__main__':
    unittest.main()
