import numpy as np
import matplotlib.pyplot as plt
import re
import glob
import pandas as pd

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

volume_indexes = []
docnum = 0
time_slice = 0 # multiple documents can occur in one time slice
topic_matrix = np.empty((num_sentilabs, num_docs, num_topics)) # holds a topic matrix for each senti-topic
for i in range(len(list_of_files)):
    for doc in docs[i]:
        for sentilab, distr in enumerate(list(filter(None, doc.splitlines()))): # enumerate brings sentiment-number
            row = np.array([float(el) for el in distr.split()])
            topic_matrix[sentilab, docnum, :] = row
        docnum = docnum + 1
    volume_indexes.append(docnum)
    time_slice = time_slice + 1 # this can be used to set vertical lines in the plot later

# print(','.join(topic_words[8]))

max_y = []

# 1st: sentiLabel, 3rd: topicLabel
series = topic_matrix[1, :, 1] # the column represents the topic evolution "over time" for one topic and sentiment
series_smooth = pd.rolling_mean(series, 2)
plt.plot(series, '.', alpha=0.3, c="g")  # '.' specifies the type of mark to use on the graph
plt.plot(series_smooth, '-', linewidth=2, c="g")
max_y.append(np.max(series))

series = topic_matrix[2, :, 1]
series_smooth = pd.rolling_mean(series, 2)
plt.plot(series, '.', alpha=0.3, c="r")
plt.plot(series_smooth, '-', linewidth=2, c="r")
max_y.append(np.max(series))

series = topic_matrix[0, :, 1]
series_smooth = pd.rolling_mean(series, 2)
plt.plot(series, '.', alpha=0.3, c="g")
plt.plot(series_smooth, '--', linewidth=2, c="b", alpha=0.3)
max_y.append(np.max(series))


plt.vlines(volume_indexes, ymin=0, ymax=np.max(max_y))
plt.ylim(0, max(max_y))

plt.title('Topic-Senti Visualization')
plt.ylabel("Sentiment share")
plt.xlabel("Time Slices")
plt.xticks(volume_indexes)

plt.xticks(volume_indexes, range(1, time_slice+1), rotation='horizontal')

plt.tight_layout()
plt.show()




