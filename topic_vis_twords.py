import matplotlib.pyplot as plt
import glob
import re

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

docs = []
num_top_words = 10 # count of the twords that shall be displayed

list_of_files = glob.glob('C:\\Users\\kmr\\Downloads\\JST-master\\JST-master\\result\\test\\t2\\*.twords')
for file_name in list_of_files:
    with open(file_name) as f:
       read_data = f.read()
       docs.append([doc for doc in filter(None, re.split(r"Label[0-9]+_Topic[0-9]+", read_data))])

twords = []
for twords_tuple in zip(*docs):
    words = [list(filter(lambda x: not is_number(x), word.split()))[:num_top_words] for word in twords_tuple] # holds twords for one specific sentilabel and topiclabel
    twords.append(words)

num_topics = 3
num_sentilabs = 3
time_slices = 5

for t in range(time_slices):
    plt.subplot(1, time_slices , t + 1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('Time Slice {}'.format(t))

    for i, word in enumerate(twords[8][t]):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=12)

plt.tight_layout()
plt.show()
