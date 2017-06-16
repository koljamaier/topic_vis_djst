import numpy as np
import matplotlib.pyplot as plt
import re
import glob

list_of_files = glob.glob('C:\\Users\\kmr\\Downloads\\JST-master\\JST-master\\result\\test\\t2\\*.others')

num_topics = num_sentilabs = num_docs = 0
for file_name in list_of_files:
    with open(file_name) as f:
       for line in f:
           if("=" in line):
               parameter, value= line.rstrip().split("=")
               if(parameter == "numSentiLabs"):
                   num_sentilabs = int(value)
               elif(parameter == "numTopics"):
                   num_topics = int(value)
               elif(parameter == "numDocs"):
                   num_docs = num_docs + int(value)

docs = []
list_of_files = glob.glob('C:\\Users\\kmr\\Downloads\\JST-master\\JST-master\\result\\test\\t2\\*.theta')

"""
.theta files represent documents by topic-proportions. This
document is represented with 3 topics (col) and 3 sentiments (rows):

Document 0
0.836732 0.144089 0.019179 
0.504217 0.196829 0.298954 
0.011114 0.561325 0.427561 
"""
for file_name in list_of_files:
    with open(file_name) as f:
       read_data = f.read()
       docs.append([doc for doc in filter(None, re.split(r"Document [0-9]+", read_data))])

docnum = 0
topic_matrix = np.empty((num_sentilabs, num_docs, num_topics)) # holds a topic matrix for each senti-topic
for i in range(len(list_of_files)):
    for doc in docs[i]:
        for sentilab, distr in enumerate(list(filter(None, doc.splitlines()))): # enumerate brings sentiment-number
            row = np.array([float(el) for el in distr.split()])
            topic_matrix[sentilab, docnum, :] = row
        docnum = docnum + 1


# get the topic words
# muss umgeschrieben werden auf .twords
with open("C:\\Users\\kmr\\mallet-2.0.8\\tmp\\topic-keys-hugo.txt") as input:
    topic_keys_lines = input.readlines()

topic_words = []
for line in topic_keys_lines:
    _, _, words = line.split('\t')  # tab-separated
    words = words.rstrip().split(' ')  # remove the trailing '\n'
    topic_words.append(words)

# print(','.join(topic_words[8]))

series = topic_matrix[1, :, 0] # the column represents the topic evolution "over time"
# print(sum(series))
plt.plot(series, '.')  # '.' specifies the type of mark to use on the graph
plt.show()




