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

    # load already preprocessed data - in this case we dont need the above functions
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

    text_tokens = []
    for doc in final_docs:
        text_tokens.append(doc.split())

    # prepare heldout document
    perp_doc = []
    with codecs.open('C:\\Users\\kmr\\Downloads\\JST-master\\JST-master\\data\\perp.dat', 'r', "utf-8") as f:  # evtl 'r', encoding="utf-8")
        read_data = f.read()
        perp_doc =[doc for doc in read_data.split("\r\n") if doc]

    perp_doc_final = []
    for doc in perp_doc:
        t = doc.split()
        t[0] = ""
        perp_doc_final.append(" ".join(t).strip())

    perp_tokens = []
    for doc in perp_doc_final:
        perp_tokens.append(doc.split())

    corpus_size = len(perp_doc_final)


    # turn tokenized documents into id/term dict
    dictionary = corpora.Dictionary(text_tokens)
    # for each count the corresponding vocab-id gets + 1
    corpus = [dictionary.doc2bow(text) for text in text_tokens]
    test_heldout = [dictionary.doc2bow(text) for text in perp_tokens]
    # num_topics > 9 fails. Open issue in pyLDAvis
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, iterations=800) # passes=20, alpha='auto' EM für alpha,alpha="symmetric"

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
    for topic in ldamodel.show_topics(num_topics=3, num_words=10):
        print(topic)

    print("Succes")