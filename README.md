# Annotations of the letters of Walter Benjamin

The project documented here was part of an university assignment presented on Sep 21, 2017 in the [*Introduction to Digital Philology*](http://www.dh.uni-leipzig.de/wo/courses/) course of [Monica Berti](https://github.com/monberti) at the University of Leipzig.


What you can find here are [POS](https://github.com/herreio/benann/blob/master/data/de_pos_wa/) and [NER](https://github.com/herreio/benann/blob/master/data/de_ner_wa/) tagged letters of Walter Benjamin as well as the [raw OCR](https://github.com/herreio/benann/blob/master/data/de_raw/) and [plain text](https://github.com/herreio/benann/blob/master/data/de_lttr/) versions of them.


The initial part of speech tagging was done with a [NLTK](http://www.nltk.org/) classifier modified for [german language](https://github.com/ptnplanet/NLTK-Contributions/blob/master/ClassifierBasedGermanTagger/ClassifierBasedGermanTagger.py) and trained on the [TIGER corpus](https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/TIGERCorpus/download/start.html), while corrections and manual tagging were done with the [WebAnno](https://webanno.github.io/webanno/) tool.


The goal was to train a named entity recognizer which is capable of identifying not only personal names and locations but also literary works and publishing actors (publishers, journals, etc.).


The source text was taken from the 1978 edition of Benjamin’s letters which is freely available at [archive.org](https://archive.org/details/GesammelteSchriftenBriefe).
