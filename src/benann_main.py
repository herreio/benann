from .BenjaminCorpus import BenCorp
from .BenjaminAnnotation import BenAnn

# bc = BenCorp()
# bc.ner_tag_stats()
# bc.write_ner_stats()

ba = BenAnn()
ba.create_ner_chunker()
# ba.write_cr_letters()

# ba.rename_wa_letters() # strukturiert aus WebAnno exportierte Dateien um

# ba.read_raw_letters()
# ba.pos_tag_letters()
# ba.write_pos_letters()

# ba.pos("Dies ist ein Versuch Dir den 'Brecht' zu erklären")


# ba.read_wa_letters()
# ba.create_ner_chunker()
# ba.save_ner_chunker()
# ba.save_ner_chunker()
# ba.ner_tagger = ba.load_ner_chunker()

# ba.ner_tag_string("Dies ist ein Versuch Dir den 'Brecht' zu erklären")

# ba.ner_tag_letters()      # fehlt bisher..


# ba.read_wa_letters()      #  exportiert von WebAnno
# ba.write_cr_letters()     #  in txt/anno/de_conll
# ba.create_corpus()        #  liest conll_letters

# print(ba.corpus.chunked_sents())
