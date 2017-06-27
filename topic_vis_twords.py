import matplotlib.pyplot as plt
import glob
import re

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

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

docs = []
num_top_words = 10 # count of twords that shall be displayed

list_of_files = glob.glob('C:\\Users\\kmr\\Downloads\\JST-master\\JST-master\\result\\test\\t2\\*.twords')
list_of_files.sort(key=natural_keys)
for file_name in list_of_files:
    with open(file_name) as f:
       read_data = f.read()
       docs.append([doc for doc in filter(None, re.split(r"Label[0-9]+_Topic[0-9]+", read_data))])

twords = []
for twords_tuple in zip(*docs): # groups twords by topic and sentiment label
    words = [list(filter(lambda x: not is_number(x), word.split()))[:num_top_words] for word in twords_tuple] # holds twords for one specific sentilabel and topiclabel
    twords.append(words)

num_topics = 3
num_sentilabs = 3
time_slices = 5
offset = 13

# chunks a list
def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

senti_topic_twords = []
for index, i in enumerate(chunks(twords, num_sentilabs)):
    senti_topic_twords.append(i) # create list for easier access (sentilabel x topiclabel)

for t in range(time_slices):
    plt.subplot(1, time_slices , t+1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('Time Slice {}'.format(t+1+offset))

    for i, word in enumerate(senti_topic_twords[2][0][t + offset]):
        plt.text(0.3, num_top_words-i-0.5, word.decode("utf-8"), fontsize=10)

plt.tight_layout()
plt.show()
