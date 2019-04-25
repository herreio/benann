import random

from nltk import NaiveBayesClassifier
from nltk.classify import accuracy
from nltk.chunk import tree2conlltags
from nltk import ClassifierBasedTagger

from .BenjaminAnnotation import BenAnn
from .BenjaminCorpus import BenCorp
from .benann_classifier import ben_features


def feature_sets_chunk():
    print("create annotation instance...")
    ba = BenAnn()
    print("read ner letters...")
    ba.read_wa_letters()
    print("process training data...")
    data = list(ba.process_ner_letters())
    random.shuffle(data)
    split_perc = 0.1
    split_size = int(len(data) * split_perc)
    train_sents, test_sents = data[split_size:], data[:split_size]
    print("create and train classifier...")
    classy = ClassifierBasedTagger(train=train_sents, feature_detector=ben_features)
    print("test classifier...")
    print(classy.evaluate(test_sents))
    return classy


def feature_sets_stts():
    print('create annotation instance...')
    ba = BenAnn()
    print('read ner letters...')
    ba.read_wa_letters()

    print('process training data...')
    data = list(ba.process_ner_letters())
    random.shuffle(data)

    sents = []
    hists = []
    for t_sent in data:
        sent = []
        hist = []
        for t_token in t_sent:
            (w, p), n = t_token
            sent.append((w, p))
            hist.append(n)
        sents.append(sent)
        hists.append(hist)

    print('create feature sets...')
    feat_sets = []
    for i, t_sent in enumerate(data):
        for j in range(len(t_sent)):
            feat_sets.append((ben_features(sents[i], j, hists[i]), hists[i][j]))
    split_perc = 0.1
    split_size = int(len(feat_sets) * split_perc)
    train = feat_sets[split_size:]
    test = feat_sets[:split_size]

    print('create and train classifier...')
    classifier = NaiveBayesClassifier.train(train)

    print('test classifier...')
    print(accuracy(classifier, test))

    classifier.show_most_informative_features()

    errors = []

    for f, t in test:
        guess = classifier.classify(f)
        if guess != t:
            errors.append((t, guess, f['word']))

    per = []
    lit = []
    loc = []
    pub = []
    o = []

    for e in errors:
        t, g, w = e
        if 'PER' in t:
            per.append(e)
        elif 'LIT' in t:
            lit.append(e)
        elif 'LOC' in t:
            loc.append(e)
        elif 'PUB' in t:
            pub.append(e)
        else:
            o.append(e)

    e_dict = {
        '_LEN': len(errors),
        'PER': per,
        'LIT': lit,
        'LOC': loc,
        'PUB': pub,
        'O': o
    }

    return e_dict


def feature_sets_universal():
    print('create corpus instance...')
    bc = BenCorp()

    print('process training data...')
    c_sents = bc.corp.chunked_sents(tagset='universal')
    t_sents = []
    for c_sent in c_sents:
        t_sents.append(tree2conlltags(c_sent))
    sents = []
    hists = []
    for t_sent in t_sents:
        sent = []
        hist = []
        for t_token in t_sent:
            (w, p, n) = t_token
            sent.append((w, p))
            hist.append(n)
        sents.append(sent)
        hists.append(hist)

    print('create feature sets...')
    feat_sets = []
    for i, t_sent in enumerate(sents):
        for j in range(len(t_sent)):
            feat_sets.append((ben_features(sents[i], j, hists[i]), hists[i][j]))
    random.shuffle(feat_sets)
    split_perc = 0.1
    split_size = int(len(feat_sets) * split_perc)
    train = feat_sets[split_size:]
    test = feat_sets[:split_size]

    print('create and train classifier...')
    classifier = NaiveBayesClassifier.train(train)

    print('test classifier...')
    print(accuracy(classifier, test))

    errors = []

    for f, t in test:
        guess = classifier.classify(f)
        if guess != t:
            errors.append((t, guess, f['word']))

    per = []
    lit = []
    loc = []
    pub = []
    o = []

    for e in errors:
        t, g, w = e
        if 'PER' in t:
            per.append(e)
        elif 'LIT' in t:
            lit.append(e)
        elif 'LOC' in t:
            loc.append(e)
        elif 'PUB' in t:
            pub.append(e)
        else:
            o.append(e)

    e_dict = {
        '_LEN': len(errors),
        'PER': per,
        'LIT': lit,
        'LOC': loc,
        'PUB': pub,
        'O': o
    }

    return e_dict
