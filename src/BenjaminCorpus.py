import os

from nltk.corpus import ConllCorpusReader
from nltk.tree import Tree


class BenCorp:
    def __init__(self):
        self._root = './data/de_ner_cr/'                          # path where to write tagged letters
        self._file_ids = []                                       # list to store fileids of conll letters
        self.num_letters = 0
        self.corp = self._init_corpus()                           # Corpus of NER tagged letters
        self.ner_stats = None

    def _init_corpus(self):
        for root, dirs, files in os.walk(self._root):
            for filename in files:
                if filename.endswith('.txt'):
                    self._file_ids.append(filename)
                    self.num_letters += 1
        corp = ConllCorpusReader(self._root,
                                 self._file_ids,
                                 ['words', 'pos', 'chunk'],
                                 chunk_types=['LIT', 'LOC', 'PER'],
                                 encoding='utf-8', tagset='de-tiger')
        return corp

    def ner_tag_stats(self):
        tagged_chunks = []
        chunks = self.corp.chunked_sents()
        for chunk in chunks:
            for word in chunk:
                if isinstance(word, Tree):
                    tagged_chunks.append(word)

        ner_stats = {}
        for tagged_chunk in tagged_chunks:
            chunk_list = tagged_chunk.pos()
            chunk_string = ''
            for i, chunk_element in enumerate(chunk_list):
                (token, pos), ner = chunk_element
                if i < len(chunk_list) - 1:
                    chunk_string += token + ' '
                else:
                    chunk_string += token
                    if ner in ner_stats:
                        tag_count, examples = ner_stats[ner]
                        tag_count += 1
                        example_count = 1
                        example_string = chunk_string
                        for k, tagged in enumerate(examples):
                            if chunk_string in tagged:
                                example_count, example_string = tagged
                                examples.pop(k)
                                example_count += 1
                                break
                        examples.append((example_count, example_string))
                        ner_stats[ner] = (tag_count, examples)
                    else:
                        ner_stats[ner] = (1, [(1, chunk_string)])

        for k in ner_stats:
            count, examples = ner_stats[k]
            sorted(examples, key=lambda x: x[1])
        self.ner_stats = ner_stats

    def write_ner_stats(self):
        if not self.ner_stats:
            self.ner_tag_stats()
        tkn_cnt = ''
        for k in self.ner_stats:
            count, examples = self.ner_stats[k]
            tkn_cnt += '--------------------\n' + k + "\t" + str(count) + "\n\n"
            for example in examples:
                ex_count, ex_token = example
                tkn_cnt += ex_token + "/" + str(ex_count) + "  "
            tkn_cnt += "\n\n"
        with open('./ner_stats.txt', 'w+') as f:
            f.write(tkn_cnt)
