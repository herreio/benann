import re
import collections

from nltk.chunk import ChunkParserI, conlltags2tree
from nltk.stem.snowball import GermanStemmer
from nltk.tag.sequential import ClassifierBasedTagger


class NamedEntityChunkParser(ChunkParserI):
    def __init__(self, train_sents=None, **kwargs):
        assert isinstance(train_sents, collections.Iterable)
        self.feature_detector = self.ben_features
        self.tagger = ClassifierBasedTagger(
            train=train_sents,
            feature_detector=self.feature_detector,
            **kwargs
        )

    def parse(self, tagged_sents):
        chunks = self.tagger.tag(tagged_sents)
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]
        return conlltags2tree(iob_triplets)

    @staticmethod
    def shape(word):
        word_shape = 'other'
        if re.match('[0-9]+([.,][0-9]*)?|[0-9]*[.,][0-9]+$', word):
            word_shape = 'number'
        elif re.compile('\W+$', re.UNICODE).match(word):
            word_shape = 'punct'
        elif re.match('[A-ZÄÖÜ][a-zäöüß]+$', word):
            word_shape = 'capitalized'
        elif re.match('[A-ZÄÖÜ]+$', word):
            word_shape = 'uppercase'
        elif re.match('[a-zäöüß]+$', word):
            word_shape = 'lowercase'
        elif re.compile("\w+", re.UNICODE).match(word):
            word_shape = 'mixedcase'
        return word_shape

    def ben_features(self, tokens, index, history):
        stemmer = GermanStemmer()

        word, pos = tokens[index]
        lemma = stemmer.stem(word)
        contains_dash = '-' in word
        contains_dot = '.' in word

        if index == 0:
            prevword = prevpos = prevlemma = None
            prevshape = prevtag = None
            prevprevword = prevprevpos = prevprevtag = None
            prevprevlemma = prevprevshape = None
        elif index == 1:
            prevword, prevpos = tokens[index - 1]
            prevlemma = stemmer.stem(prevword)
            prevtag = history[index - 1]
            prevshape = self.shape(prevword)
            prevprevword = prevprevpos = prevprevtag = None
            prevprevlemma = prevprevshape = None
        else:
            prevword, prevpos = tokens[index - 1]
            prevlemma = stemmer.stem(prevword)
            prevshape = self.shape(prevword)
            prevtag = history[index - 1]
            prevprevword, prevprevpos = tokens[index - 2]
            prevprevlemma = stemmer.stem(prevprevword)
            prevprevshape = self.shape(prevprevword)
            prevprevtag = history[index - 2]
        if index == len(tokens) - 1:
            nextword = nextpos = None
            nextlemma = nextshape = None
            nextnextword = nextnextpos = None
            nextnextlemma = nextnextshape = None
        elif index == len(tokens) - 2:
            nextword, nextpos = tokens[index + 1]
            nextlemma = stemmer.stem(nextword)
            nextshape = self.shape(nextword)
            nextnextword = nextnextpos = None
            nextnextlemma = nextnextshape = None
        else:
            nextword, nextpos = tokens[index + 1]
            nextlemma = stemmer.stem(nextword)
            nextshape = self.shape(nextword)
            nextnextword, nextnextpos = tokens[index + 2]
            nextnextlemma = stemmer.stem(nextnextword)
            nextnextshape = self.shape(nextnextword)

        features = {
            'word': word,
            'pos': pos,
            'lemma': lemma,
            'shape': self.shape(word),
            # 'wordlen': len(word),

            # 'prefix1': word[:1],
            # 'prefix2': word[:1],
            # 'prefix3': word[:3],
            # 'suffix4': word[-4:],
            # 'suffix3': word[-3:],
            # 'suffix2': word[-2:],

            'prevword': prevword,
            'prevpos': prevpos,
            'prevtag': prevtag,
            'prevlemma': prevlemma,
            'prevshape': prevshape,

            'prevprevword': prevprevword,
            'prevprevpos': prevprevpos,
            'prevprevtag': prevprevtag,
            'prevprevlemma': prevprevlemma,
            'prevprevshape': prevprevshape,

            'nextword': nextword,
            'nextpos': nextpos,
            'nextshape': nextshape,
            'nextlemma': nextlemma,

            'nextnextword': nextnextword,
            'nextnextpos': nextnextpos,
            'nextnextshape': nextnextshape,
            'nextnextlemma': nextnextlemma,

            # nltk chunk tagger
            # 'word+nextpos': '{0}+{1}'.format(word, nextpos),
            # 'word+prevpos': '{0}+{1}'.format(word, prevpos),
            # 'word+prevtag': '{0}+{1}'.format(word, prevtag),
            # 'word+nexttag': '{0}+{1}'.format(word, nexttag),
            # 'pos+prevtag': '{0}+{1}'.format(pos, prevtag),
            # 'shape+prevtag': '{0}+{1}'.format(prevshape, prevtag),

            # german tagger
            # 'word+prevword': '{0}+{1}'.format(word, prevword),
            # 'word+prevprevtag': '{0}+{1}'.format(word, prevprevtag),

            # 'containsdash': contains_dash,
            # 'containsdot': contains_dot,
        }

        return features

    def eval_chunker(self, test_samples):
        score = self.evaluate([conlltags2tree([(w, t, iob) for (w, t), iob in iobs]) for iobs in test_samples])
        return score
