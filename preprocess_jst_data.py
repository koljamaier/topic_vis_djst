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
This script concatenates all .dat files for djst-training
to one file for JST training
"""

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


list_of_files = glob.glob('C:\\Users\\kmr\\Downloads\\JST-master\\JST-master\\data\\temp\\docs_for_python_parser\\pos_ext\\*.dat')
list_of_files.sort(key=natural_keys)

#f = codecs.open("test", "r", "utf-8")
epoch_docs = []
for file_name in list_of_files:
    with codecs.open(file_name) as f: # evtl. ,'r',"utf-8"
       read_data = f.read()
       epoch_docs.append([doc for doc in read_data.split("\r\n") if doc])

with codecs.open("C:\\Users\\kmr\\Downloads\\JST-master\\JST-master\\data\\summary.dat", "w") as f:
    for epoch_doc in epoch_docs:
        for doc in epoch_doc:
            f.write(doc)
        f.write("\n")
