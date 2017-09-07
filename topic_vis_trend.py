import numpy as np
import matplotlib.pyplot as plt
import re
import glob
import pandas as pd

"""
This script creates trend-plots from the topic
model outputs theta and pi
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import glob
import pandas as pd

"""
This script creates djst trend-plots
"""

def atoi(text):
    return int(text) if text.isdigit() else text


# This is needed to sort the .twords files
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

result_path = "C:\\Users\\kmr\\Downloads\\JST-master\\JST-master\\result\\"

#file_path = "vw\\vw_half1_mixed\\3topics_lag"
#file_path = "vw\\vw_half1_pos\\3topics_lag"
#file_path = "vw\\vw_half1_pos_half2_neg\\3topics_lag"
#file_path = "vw\\vw_2neg_peaks\\3topics_lag_corr"
file_path = "vw\\vw_1neg_peaks\\3topics_lag_corr"
#file_path = "vw\\vw_orig\\3topics_lag_corr"
#file_path = "vw\\vw_orig\\3topics_lag"

topics = 3
#bei twitter hoch, bei VW niedrig
smoothness = 3

# list_of_files = glob.glob('C:\\Users\\kmr\\Downloads\\JST-master\\JST-master\\result\\test\\bla\\*.others')
list_of_files = glob.glob(result_path+ file_path+"\\*.others")
list_of_files.sort(key=natural_keys)

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

list_of_files_pi = glob.glob(result_path+ file_path+"\\*.pi")
list_of_files_pi.sort(key=natural_keys)

list_of_files_theta = glob.glob(result_path+ file_path + "\\*.theta")
list_of_files_theta.sort(key=natural_keys)




"""
.theta files represent documents by topic-proportions. This
document is represented with 3 topics (col) and 3 sentiments (rows):

.pi files represent docs by senti-proportions. The following
document ist represented with 3 sentiments (cols):

d_0 d0 0.611984 0.248729 0.139287

"""
joint_matrix = np.empty((num_sentilabs, len(list_of_files_theta), num_topics)) # holds senti-topic info over all epochs
list_of_files = zip(list_of_files_pi,list_of_files_theta) # holds name for pi and theta file
docs_pi = []
docs_theta = []
epoch = 0
volume_indexes = []
for file_name in list_of_files: # file_name for one epoch
    with open(file_name[0]) as f: # for pi_file
        read_data = f.read()
        #docs_pi.append(read_data.splitlines())
        docs_pi = [doc for doc in filter(None, re.split(r"\n", read_data))]

    with open(file_name[1]) as f:  # for theta_file
       read_data = f.read()
       docs_theta = [doc for doc in filter(None, re.split(r"Document [0-9]+", read_data))]

    num_docs1 = len(docs_pi)
    docnum = 0
    # parse theta for this epoch
    topic_matrix = np.empty((num_sentilabs, num_docs1, num_topics))  # acces: 1# sentilab, 2# doc_num 3# topic_num
    for doc in docs_theta:
        for sentilab, distr in enumerate(list(filter(None, doc.splitlines()))):  # enumerate brings sentiment-number
            row = np.array([float(el) for el in distr.split()])
            topic_matrix[sentilab, docnum, :] = row
        docnum = docnum + 1

    # parse pi for this epoch
    docnum = 0
    senti_matrix = np.empty((num_sentilabs, num_docs1))
    for doc in docs_pi:
        for sentilab in range(num_sentilabs):
            row = np.array([float(el) for el in doc.split()[2:]]) # discards document description
            senti_matrix[:, docnum] = row
        docnum = docnum + 1

    # multiplicate theta and pi
    # write result in joint_matrix for this epoch
    for d in range(len(docs_pi)):
        for t in range(num_topics):
            for l in range(num_sentilabs):
                joint_matrix[l, epoch, t] += (topic_matrix[l,d,t]*senti_matrix[l,d])/(num_docs1)
    volume_indexes.append(epoch) # für den plot
    epoch = epoch + 1



# test for: 3topics_pos: topic=0 smoothness=3
max_y = []



for topic in range(topics):
    # 1st: sentiLabel, 3rd: topicLabel
    series = joint_matrix[1, :, topic]# the column represents the topic evolution "over time" for one topic and sentiment
    series_smooth = pd.Series(series).rolling(window=smoothness).mean()
    #plt.plot(series, '.', alpha=0.9, c="g")  # '.' specifies the type of mark to use on the graph
    plt.plot(series_smooth, '-', linewidth=2, c="g")
    max_y.append(np.max(series))

    series = joint_matrix[2, :, topic]# the column represents the topic evolution "over time" for one topic and sentiment
    series_smooth = pd.Series(series).rolling(window=smoothness).mean()
    #plt.plot(series, '.', alpha=0.9, c="r")  # '.' specifies the type of mark to use on the graph
    plt.plot(series_smooth, '-', linewidth=2, c="r")
    max_y.append(np.max(series))

    series = joint_matrix[0, :,topic]  # the column represents the topic evolution "over time" for one topic and sentiment
    series_smooth = pd.Series(series).rolling(window=smoothness).mean()
    # plt.plot(series, '.', alpha=0.3, c="b")  # '.' specifies the type of mark to use on the graph
    #plt.plot(series_smooth, '--', linewidth=2, c="b")
    #max_y.append(np.max(series))



    #series = topic_matrix[1, :, topic]
    #series_smooth = pd.rolling_mean(series, smoothness)
    #plt.plot(series, '.', alpha=0.3, c="b")
    #plt.plot(series_smooth, '--', linewidth=2, c="b", alpha=0.3)
    #max_y.append(np.max(series))


    plt.vlines(volume_indexes, ymin=0, ymax=np.max(max_y))
    plt.ylim(0, max(max_y))

    plt.title('Topic-Senti Visualization for '+file_path)
    plt.ylabel("Sentiment share")
    plt.xlabel("Time Slices")
    plt.xticks(volume_indexes)
    plt.xticks(volume_indexes, range(1, epoch+1), rotation='horizontal')

    plt.tight_layout()
    plt.show()


def atoi(text):
    return int(text) if text.isdigit() else text


# This is needed to sort the .twords files
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

result_path = "C:\\Users\\kmr\\Downloads\\JST-master\\JST-master\\result\\"

#file_path = "vw\\vw_half1_mixed\\3topics_lag"
#file_path = "vw\\vw_half1_pos\\3topics_lag"
#file_path = "vw\\vw_half1_pos_half2_neg\\3topics_lag"
#file_path = "vw\\vw_2neg_peaks\\3topics_lag_corr"
file_path = "vw\\vw_1neg_peaks\\3topics_lag_corr"
#file_path = "vw\\vw_orig\\3topics_lag_corr"
#file_path = "vw\\vw_orig\\3topics_lag"

topics = 3
#bei twitter hoch, bei VW niedrig
smoothness = 3

# list_of_files = glob.glob('C:\\Users\\kmr\\Downloads\\JST-master\\JST-master\\result\\test\\bla\\*.others')
list_of_files = glob.glob(result_path+ file_path+"\\*.others")
list_of_files.sort(key=natural_keys)

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

list_of_files_pi = glob.glob(result_path+ file_path+"\\*.pi")
list_of_files_pi.sort(key=natural_keys)

list_of_files_theta = glob.glob(result_path+ file_path + "\\*.theta")
list_of_files_theta.sort(key=natural_keys)




"""
.theta files represent documents by topic-proportions. This
document is represented with 3 topics (col) and 3 sentiments (rows):

.pi files represent docs by senti-proportions. The following
document ist represented with 3 sentiments (cols):

d_0 d0 0.611984 0.248729 0.139287

"""
joint_matrix = np.empty((num_sentilabs, len(list_of_files_theta), num_topics)) # holds senti-topic info over all epochs
list_of_files = zip(list_of_files_pi,list_of_files_theta) # holds name for pi and theta file
docs_pi = []
docs_theta = []
epoch = 0
volume_indexes = []
for file_name in list_of_files: # file_name for one epoch
    with open(file_name[0]) as f: # for pi_file
        read_data = f.read()
        #docs_pi.append(read_data.splitlines())
        docs_pi = [doc for doc in filter(None, re.split(r"\n", read_data))]

    with open(file_name[1]) as f:  # for theta_file
       read_data = f.read()
       docs_theta = [doc for doc in filter(None, re.split(r"Document [0-9]+", read_data))]

    num_docs1 = len(docs_pi)
    docnum = 0
    # parse theta for this epoch
    topic_matrix = np.empty((num_sentilabs, num_docs1, num_topics))  # acces: 1# sentilab, 2# doc_num 3# topic_num
    for doc in docs_theta:
        for sentilab, distr in enumerate(list(filter(None, doc.splitlines()))):  # enumerate brings sentiment-number
            row = np.array([float(el) for el in distr.split()])
            topic_matrix[sentilab, docnum, :] = row
        docnum = docnum + 1

    # parse pi for this epoch
    docnum = 0
    senti_matrix = np.empty((num_sentilabs, num_docs1))
    for doc in docs_pi:
        for sentilab in range(num_sentilabs):
            row = np.array([float(el) for el in doc.split()[2:]]) # discards document description
            senti_matrix[:, docnum] = row
        docnum = docnum + 1

    # multiplicate theta and pi
    # write result in joint_matrix for this epoch
    for d in range(len(docs_pi)):
        for t in range(num_topics):
            for l in range(num_sentilabs):
                joint_matrix[l, epoch, t] += (topic_matrix[l,d,t]*senti_matrix[l,d])/(num_docs1)
    volume_indexes.append(epoch) # für den plot
    epoch = epoch + 1



# test for: 3topics_pos: topic=0 smoothness=3
max_y = []



for topic in range(topics):
    # 1st: sentiLabel, 3rd: topicLabel
    series = joint_matrix[1, :, topic]# the column represents the topic evolution "over time" for one topic and sentiment
    series_smooth = pd.Series(series).rolling(window=smoothness).mean()
    #plt.plot(series, '.', alpha=0.9, c="g")  # '.' specifies the type of mark to use on the graph
    plt.plot(series_smooth, '-', linewidth=2, c="g")
    max_y.append(np.max(series))

    series = joint_matrix[2, :, topic]# the column represents the topic evolution "over time" for one topic and sentiment
    series_smooth = pd.Series(series).rolling(window=smoothness).mean()
    #plt.plot(series, '.', alpha=0.9, c="r")  # '.' specifies the type of mark to use on the graph
    plt.plot(series_smooth, '-', linewidth=2, c="r")
    max_y.append(np.max(series))

    series = joint_matrix[0, :,topic]  # the column represents the topic evolution "over time" for one topic and sentiment
    series_smooth = pd.Series(series).rolling(window=smoothness).mean()
    # plt.plot(series, '.', alpha=0.3, c="b")  # '.' specifies the type of mark to use on the graph
    #plt.plot(series_smooth, '--', linewidth=2, c="b")
    #max_y.append(np.max(series))

    #series = topic_matrix[1, :, topic]
    #series_smooth = pd.rolling_mean(series, smoothness)
    #plt.plot(series, '.', alpha=0.3, c="b")
    #plt.plot(series_smooth, '--', linewidth=2, c="b", alpha=0.3)
    #max_y.append(np.max(series))

    plt.vlines(volume_indexes, ymin=0, ymax=np.max(max_y))
    plt.ylim(0, max(max_y))

    plt.title('Topic-Senti Visualization for '+file_path)
    plt.ylabel("Sentiment share")
    plt.xlabel("Time Slices")
    plt.xticks(volume_indexes)
    plt.xticks(volume_indexes, range(1, epoch+1), rotation='horizontal')

    plt.tight_layout()
    plt.show()
