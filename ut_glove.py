import unittest
import word_embedding


class GloVeUT(unittest.TestCase):
    def test_wmd(self):
        we = word_embedding.WordEmbedding("../glove.6B/glove.6B.50d.txt")
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