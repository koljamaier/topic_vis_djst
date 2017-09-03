from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import pyLDAvis.gensim
import glob
import re
import codecs
import numpy as np
"""
This script creates a LDA model (via gensim).
The results will be displayed via pyLDAvis.
The same data as for the djst model is used.
"""
# https://groups.google.com/forum/#!topic/gensim/TpuYRxhyIOc
# Für die Auswertung der einzelnen gensim topics https://stackoverflow.com/questions/15067734/lda-model-generates-different-topics-everytime-i-train-on-the-same-corpus
if __name__ == "__main__":

    SEED = 42
    np.random.seed(SEED)
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [ atoi(c) for c in re.split('(\d+)', text) ]


    def clean_doc(text):
        # clean and tokenize document string
        raw = text.lower()
        tokens = tokenizer.tokenize(raw)

        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]

        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

        # add tokens to list
        texts.append(stemmed_tokens)
        return len(stemmed_tokens)

    list_of_files = glob.glob('C:\\Users\\kmr\\Downloads\\JST-master\\JST-master\\data\\temp\\docs_for_python_parser\\pos_ext\\*.dat')
    list_of_files.sort(key=natural_keys)

    #f = codecs.open("test", "r", "utf-8")
    epoch_docs = []
    for file_name in list_of_files:
        with codecs.open(file_name) as f: # evtl. ,'r',"utf-8"
           read_data = f.read()
           epoch_docs.append([doc for doc in read_data.split("\r\n") if doc])

    final_docs = []
    for docs in epoch_docs:
        for doc in docs:
            t = doc.split()
            t[0]="" # discard d0...
            final_docs.append(" ".join(t).strip())


    perp_doc = []
    with codecs.open('C:\\Users\\kmr\\Downloads\\JST-master\\JST-master\\data\\perp.dat', 'r', "utf-8") as f:  # evtl 'r', encoding="utf-8")
        read_data = f.read()
        perp_doc =[doc for doc in read_data.split("\r\n") if doc]

    perp_doc_final = []
    for doc in perp_doc:
        t = doc.split()
        t[0] = ""
        perp_doc_final.append(" ".join(t).strip())

    corpus_size = len(perp_doc_final)


    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = set(stopwords.words('english'))
    p_stemmer = PorterStemmer()

    # doc_set = final_docs
    doc_set = []
    # list for tokenized documents in loop
    texts = []

    for doc in final_docs:
        texts.append(doc.split())

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)

    # convert tokenized documents into a document-term matrix (fuer jeden count des documents bekommt die entsprechende vocab-id +1) Siehe gensim docs
    corpus = [dictionary.doc2bow(text) for text in texts]

    # generate LDA model
    # num_topics > 9 fail. Open issue in pyLDAvis
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, iterations=800) # passes=20, alpha='auto' EM für alpha,alpha="symmetric"

    print(texts[len(texts)-1])
    heldout = [texts[len(texts)-1], texts[len(texts)-2]]
    heldout2 = [corpus[1], corpus[2]]

    perp_texts = []
    for doc in perp_doc_final:
        perp_texts.append(doc.split())

    test_heldout = [dictionary.doc2bow(text) for text in perp_texts]

    #print(ldamodel.log_perplexity(corpus[1]))
    perplex = ldamodel.bound(test_heldout)
    print(perplex)
    print(ldamodel.log_perplexity(test_heldout))
    per_word_perplex = np.exp2(-perplex / corpus_size)
    print(per_word_perplex)
    ## Interactive visualisation
    vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
    #pyLDAvis.display(vis)
    pyLDAvis.save_html(vis, "vw_lda_3topics.html")
    print("Succes")