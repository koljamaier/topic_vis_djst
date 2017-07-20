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

list_of_files = glob.glob('C:\\Users\\kmr\\Downloads\\JST-master\\JST-master\\result\\test\\brexit\\*.twords')
list_of_files.sort(key=natural_keys)
for file_name in list_of_files:
    with open(file_name) as f:
       read_data = f.read()
       docs.append([doc for doc in filter(None, re.split(r"Label[0-9]+_Topic[0-9]+", read_data))])

twords = []
for twords_tuple in zip(*docs): # groups twords by topic and sentiment label
    words = [list(filter(lambda x: not is_number(x), word.split()))[:num_top_words] for word in twords_tuple] # holds twords for one specific sentilabel and topiclabel
    twords.append(words)

num_topics = 8
num_sentilabs = 3
time_slices = 5
offset = 0

# chunks a list
def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

senti_topic_twords = []
for index, i in enumerate(chunks(twords, num_topics)):
    senti_topic_twords.append(i) # create list for easier access (sentilabel x topiclabel)


cloud_words = []
# displays 5 topics for a given sentiment over time
for z in range(11):
    #for k in range(num_topics): # topics
    for t in range(time_slices):
        plt.subplot(1, time_slices , t+1)  # plot numbering starts with 1
        plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
        plt.xticks([])
        plt.yticks([])
        plt.title('Time Slice {}'.format(t+1+offset))
        #1st senti, 2nd topic-label
        for i, word in enumerate(senti_topic_twords[2][2][t + offset]):
            plt.text(0.3, num_top_words-i-0.5, word.decode("utf-8"), fontsize=10)
            cloud_words.append(word)
    plt.tight_layout()
    plt.show()
    plt.close()
    offset = offset + 5
import matplotlib.pyplot as plt
from wordcloud import WordCloud

wordcloud = WordCloud(background_color="white", max_words=80, collocations=False).generate(" ".join(cloud_words))

plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")
plt.show()
plt.close()


print("done")