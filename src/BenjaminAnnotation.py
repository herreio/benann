import os
import pickle
import random
import re

from nltk import sent_tokenize
from nltk.corpus import ConllCorpusReader
from nltk.tokenize import word_tokenize

from .PartOfSpeechGermanTagger import PartOfSpeechGermanTagger
from .NamedEntityChunkParser import NamedEntityChunkParser


class BenAnn:
    def __init__(self, raw_root = './data/de_lttr/'):
        self.raw_root = raw_root                                        # path where to read raw letters
        self.raw_letters = []                                           # list to store read letters
        self.raw_file_ids = []                                          # list to store fileids of read letters
        self.num_raw_letters = 0                                        # number of letters read

        self.pos_tagger_path = './pickle/benann_pos_classifier_data.pickle'      # path where to store/find PoS tagger
        self.pos_tagger = None # self.load_pos_tagger()                          # PoS tagger trained with Tiger corpus
        self.pos_score = -1                                             # score of trained PoS tagger
        self.pos_root = './data/de_out/'                                # path where to write tagged letters
        self.pos_letters = []                                           # list to store tagged letters
        self.num_pos_letters = 0                                        # number of tagged letters

        self.wa_root = './data/de_ner_wa/'                              # path where to read NER tagged letters
        self.wa_xp_root = './data/de_ner_xp/'                           # path where to read letters exported from WA
        self.wa_letters = []                                            # list to store read letters
        self.wa_file_ids = []                                           # list to store fileids of NER tagged letters
        self.num_wa_letters = 0

        self.ner_cr_root = './data/de_ner_cr/'                          # path where to write tagged letters

        self.ner_letters = []                                           # list to store tagged letters
        self.num_ner_letters = 0                                        # number of letters tagged

        self.ner_chunker_path = './pickle/benann_ner_classifier_data.pickle'     # path where to store/find NER tagger
        self.ner_chunker = None # self.load_ner_chunker()                        # NER chunk parser
        self.ner_tagger = None # self.ner_chunker.tagger                         # NER tagger trained with tagged letters
        self.ner_score = -1                                             # score of trained NER tagger

        self.corpus = None                                              # Corpus of NER tagged letters

    ############################################
    # read letters after they have been ocr-ed #
    ############################################

    def read_raw_letters(self):
        for root, dirs, files in os.walk(self.raw_root):
            for filename in files:
                if filename.endswith('.txt'):
                    self.raw_file_ids.append(filename)
                    filepath = os.path.join(root, filename)
                    letter = self.read_raw_letter(filepath)
                    self.raw_letters.append(letter)
                    self.num_raw_letters += 1

    @staticmethod
    def read_raw_letter(filepath):
        with open(filepath, 'rb') as f:
            raw_lines = f.readlines()
            paragraphs = []
            for i, raw_line in enumerate(raw_lines):
                if i:
                    uni_line = raw_line.decode('utf-8').strip()
                    paragraphs.append(uni_line)
                else:   # skip first line
                    continue
        return paragraphs

    #######################################################################
    # create, save and load PoS-tagger with tiger corpus as training data #
    #######################################################################

    def create_pos_tagger(self):
        corp = ConllCorpusReader('./corpus',
                                 'tiger_release_aug07.corrected.16012013.conll09',
                                 ['ignore', 'words', 'ignore', 'ignore', 'pos'], encoding='utf-8',
                                 tagset='de-tiger')
        tagged_sents = list(corp.tagged_sents())
        random.shuffle(tagged_sents)
        split_perc = 0.1
        split_size = int(len(tagged_sents) * split_perc)
        train_sents, test_sents = tagged_sents[split_size:], tagged_sents[:split_size]
        self.pos_tagger = PartOfSpeechGermanTagger(train=train_sents)
        self.pos_score = self.pos_tagger.evaluate(test_sents)
        print(self.pos_score)

    def save_pos_tagger(self):
        if self.pos_tagger:
            with open(self.pos_tagger_path, 'wb') as f:
                pickle.dump(self.pos_tagger, f, protocol=2)
        else:
            print("first you have to create a PoS tagger!")

    def load_pos_tagger(self):
        with open(self.pos_tagger_path, 'rb') as f:
            tagger = pickle.load(f)
        return tagger

    ##################################
    # method to PoS-tag given string #
    ##################################

    def pos_tag_string(self, words):
        tokens = []
        sents = sent_tokenize(words, language='german')
        for sent in sents:
            words = word_tokenize(sent, language='german')
            for word in words:
                tokens.append(word)
        return self.pos_tagger.tag(tokens)

    ###################################
    # tag raw letters with PoS tagger #
    ###################################

    def pos_tag_letters(self):
        if self.raw_letters:
            for letter in self.raw_letters:
                tagged_letter = self.pos_tag_letter(letter)
                self.pos_letters.append(tagged_letter)
                self.num_pos_letters += 1
        else:
            print('first you have to read the raw letters')

    def pos_tag_letter(self, letter):
        tokens = []
        for paragraph in letter:
            sentences = sent_tokenize(paragraph, language='german')
            for sentence in sentences:
                words = word_tokenize(sentence, language='german')
                for word in words:
                    tokens.append(word)
        tagged_letter = self.pos_tagger.tag(tokens)
        return tagged_letter

    def pos_retag_letter(self, filename):
        letter = self.read_raw_letter(self.raw_root+filename)
        tagged_letter = self.pos_tag_letter(letter)
        self.write_pos_letter(self.pos_root+filename, tagged_letter)

    #################################################
    # write letters after they have been PoS-tagged #
    #################################################

    def write_pos_letters(self):
        if self.pos_letters:
            for filenum, letter in enumerate(self.pos_letters):
                filepath = self.pos_root + self.raw_file_ids[filenum]
                self.write_pos_letter(filepath, letter)
        else:
            print('first you have to read/tag the raw letters')

    @staticmethod
    def write_pos_letter(filepath, tagged_letter):
        with open(filepath, 'w+') as f:
            for token in tagged_letter:
                word, pos = token
                if pos == '$.' and word != ';' and word != ':':
                    out = word + " " + pos + ' O' + "\n\n"
                else:
                    out = word + " " + pos + ' O' + "\n"
                f.write(out)

    ############################################################
    # read in letters after they have been manually NER-tagged #
    ############################################################

    def rename_wa_letters(self):
        dir_names = []
        for root, dirs, files in os.walk(self.wa_xp_root):
            for subdir in dirs:
                dir_name = subdir
                dir_names.append(dir_name)
        for i, dir_name in enumerate(dir_names):
            with open(self.wa_xp_root+dir_name + "/dh55vita.tsv", 'r') as f1:
                content = f1.read()
            with open(self.wa_root+dir_name.split(".")[0] + ".tsv", 'w+') as f2:
                f2.write(content)

    def read_wa_letters(self):
        for root, dirs, files in os.walk(self.wa_root):
            for filename in files:
                if filename.endswith(".tsv"):
                    self.wa_file_ids.append(filename)
                    filepath = os.path.join(root, filename)
                    letter = self.read_wa_letter(filepath)
                    self.wa_letters.append(letter)
                    self.num_wa_letters += 1

    @staticmethod
    def read_wa_letter(filepath):
        with open(filepath, 'rb') as file:
            content = file.read().decode('utf-8').strip()
            letter = content.split('\n\n')
        return letter

    #########################################################################################
    # helper methods to create training data for NER chunker out of manually tagged letters #
    #########################################################################################

    def process_ner_letters(self):
        if not self.wa_letters:
            self.read_wa_letters()
        for tagged_letter in self.wa_letters:
            for tagged_sentence in tagged_letter:
                sent = tagged_sentence.split("\t")
                if len(sent) > 1:
                    sent_tagged_tokens = []
                    for i in range(2, len(sent), 5):
                        token = sent[i]
                        pos = sent[i+1]
                        ner = sent[i+2]
                        if ner == '_':
                            ner = 'O'
                        elif 'deriv' in ner or 'part' in ner or 'ref' in ner:
                            ner = 'O'
                        else:
                            p = re.compile('[a-zA-Z+]+')
                            ner = p.search(ner).group(0)
                        sent_tagged_tokens.append((token, pos, ner))
                    conll_tokens = self.to_conll_iob(sent_tagged_tokens)
                    yield [((w, t), iob) for w, t, iob in conll_tokens]
                else:
                    continue

    @staticmethod
    def to_conll_iob(annotated_sentence):
        proper_iob_tokens = []
        for idx, annotated_token in enumerate(annotated_sentence):
            token, pos, ner = annotated_token
            if ner != 'O':
                if idx == 0:
                    ner = "B-" + ner
                elif annotated_sentence[idx - 1][2] == ner:
                    ner = "I-" + ner
                else:
                    ner = "B-" + ner
            proper_iob_tokens.append((token, pos, ner))
        return proper_iob_tokens

    ###################################################################################
    # create, save and load NER chunker with manually tagged letters as training data #
    ###################################################################################

    def create_ner_chunker(self):
        if not self.wa_letters:
            print("read ner letters...")
            self.read_wa_letters()
        print("process training data...")
        reader = self.process_ner_letters()
        data = list(reader)
        random.shuffle(data)
        split_perc = 0.1
        split_size = int(len(data) * split_perc)
        train_sents, test_sents = data[split_size:], data[:split_size]
        print("create and train classifier...")
        self.ner_chunker = NamedEntityChunkParser(train_sents)
        self.ner_tagger = self.ner_chunker.tagger
        print("evaluate classifer...")
        self.ner_score = self.ner_chunker.eval_chunker(test_sents)
        print(self.ner_score)

    def save_ner_chunker(self):
        if self.ner_chunker:
            with open(self.ner_chunker_path, 'wb') as f:
                pickle.dump(self.ner_chunker, f, protocol=2)
        else:
            print("first you have to create a NER chunker!")

    def load_ner_chunker(self):
        with open(self.ner_chunker_path, 'rb') as f:
            tagger = pickle.load(f)
        return tagger

    ##################################
    # method to NER-tag given string #
    ##################################

    def ner_tag_string(self, words, pos=False):
        if not pos:
            pos_tagged_words = self.pos_tag_string(words)
            ner_tagged_words = self.ner_tagger.tag(pos_tagged_words)
            return ner_tagged_words
        else:
            ner_tagged_words = self.ner_tagger.tag(words)
            return ner_tagged_words

    ##############################
    # NER-tag PoS-tagged letters #
    ##############################

    def ner_tag_letter(self, filepath="", letter=None):
        if letter and filepath == "":
            pos_tagged_letter = self.pos_tag_letter(letter)
            ner_tagged_letter = self.ner_tagger.tag(pos_tagged_letter)
            self.ner_letters.append(ner_tagged_letter)
            self.num_ner_letters += 1
        elif filepath != "" and not letter:
            with open(filepath, 'rb') as file:
                raw_lines = file.readlines()
                paragraphs = []
                for raw_line in raw_lines:
                    uni_line = raw_line.decode('utf-8').strip()
                    paragraphs.append(uni_line)
            pos_tagged_letter = self.pos_tag_letter(paragraphs)
            ner_tagged_letter = self.ner_tagger.tag(pos_tagged_letter)
            self.ner_letters.append(ner_tagged_letter)
            self.num_ner_letters += 1
        else:
            print("either give path where to read raw letter or pass PoS-tagged letter as parameter!")

    ############################################
    # write NER-tagged letters in CoNLL fromat #
    ############################################

    def write_cr_letters(self):
        cr_letters = self.format_ner_letters()
        for i, cr_letter in enumerate(cr_letters):
            filename = self.wa_file_ids[i].split(".")[0] + ".txt"
            filepath = self.ner_cr_root + filename
            with open(filepath, 'w+') as f:
                for sentence in cr_letter:
                    for token in sentence:
                        word, pos, ner = token
                        if pos == '$.' and word != ';' and word != ':':
                            out = word + " " + pos + " " + ner + " \n\n"
                        else:
                            out = word + " " + pos + " " + ner + "\n"
                        f.write(out)

    ################################################################
    # helper method for writing NER-tagged letters in CoNLL format #
    ################################################################

    def format_ner_letters(self):
        if not self.wa_letters:
            self.read_wa_letters()
        cr_letters = []
        for tagged_letter in self.wa_letters:
            cr_letter = []
            for tagged_sentence in tagged_letter:
                sent = tagged_sentence.split("\t")
                if len(sent) > 1:
                    sent_tokens = []
                    for i in range(2, len(sent), 5):
                        token = sent[i]
                        pos = sent[i+1]
                        ner = sent[i+2]
                        if ner == '_':
                            ner = 'O'
                        else:
                            p = re.compile('[a-zA-Z+]+')
                            ner = p.search(ner).group(0)
                            # pos = 'NE'
                        token_pos_ner = (token, pos, ner)
                        sent_tokens.append(token_pos_ner)
                    cr_sent = self.to_conll_iob(sent_tokens)
                    cr_letter.append(cr_sent)
            cr_letters.append(cr_letter)
        return cr_letters
