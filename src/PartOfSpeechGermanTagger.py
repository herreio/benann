import re
from nltk.tag.sequential import ClassifierBasedTagger


class PartOfSpeechGermanTagger(ClassifierBasedTagger):

    def feature_detector(self, tokens, index, history):

        word = tokens[index]
        if index == 0:
            prevword = prevprevword = None
            prevtag = prevprevtag = None
        elif index == 1:
            prevword = tokens[index-1]  # Note: no lowercase
            prevprevword = None
            prevtag = history[index-1]
            prevprevtag = None
        else:
            prevword = tokens[index-1]
            prevprevword = tokens[index-2]
            prevtag = history[index-1]
            prevprevtag = history[index-2]

        if re.match('[0-9]+([.,][0-9]*)?|[0-9]*[.,][0-9]+$', word):
            shape = 'number'
        elif re.compile('\W+$', re.UNICODE).match(word):
            shape = 'punct'
        elif re.match('([A-ZÄÖÜ]+[a-zäöüß]*-?)+$', word):
            shape = 'upcase'
        elif re.match('[a-zäöüß]+', word):
            shape = 'downcase'
        elif re.compile("\w+", re.UNICODE).match(word):
            shape = 'mixedcase'
        else:
            shape = 'other'

        features = {
            'prevtag': prevtag,
            'prevprevtag': prevprevtag,
            'word': word,
            'word.lower': word.lower(),
            'suffix3': word.lower()[-3:],
            'preffix1': word[:1],
            'prevprevword': prevprevword,
            'prevword': prevword,
            'prevtag+word': '%s+%s' % (prevtag, word),
            'prevprevtag+word': '%s+%s' % (prevprevtag, word),
            'prevword+word': '%s+%s' % (prevword, word),
            'shape': shape
            }
        return features
