import pyLDAvis
import codecs

"""
Creates pyLDAvis for JST
"""

if __name__ == "__main__":
    sentis = ["","2","1","0"]
    for senti in sentis:
        result_dir = "C:\\Users\\kmr\\Downloads\\JST-master\\JST-master\\result\\vw\\JST\\3topics\\"
        doc_lengths = []
        #senti = "2"
        filename = result_dir+"doc_lengths.txt"
        with codecs.open(filename,"r", "utf-8") as f:
           read_data = f.read()
           doc_lengths = [int(line) for line in read_data.split()]

        doc_topic_dists = []
        filename = result_dir+"doc_topic_dists_theta"+senti+".txt"
        with codecs.open(filename, "r","utf-8") as f:
            read_data = f.read()
            doc_topic_dists.append([line.split() for line in read_data.splitlines()])

        doc_topic_dists = [item for sublist in doc_topic_dists for item in sublist]
        doc_topic_dists = [list(map(float, doctop)) for doctop in doc_topic_dists]

        topic_term_dists = []
        filename = result_dir+"topic_term_dists_phi"+senti+".txt"
        with codecs.open(filename, "r","utf-8") as f:
           read_data = f.read()
           topic_term_dists.append([line.split() for line in read_data.splitlines()])

        topic_term_dists = [item for sublist in topic_term_dists for item in sublist]
        topic_term_dists = [list(map(float, doctop)) for doctop in topic_term_dists]

        vocab = []
        term_frequency = []
        filename = result_dir+"vocab_term_frequency"+senti+".txt"
        with codecs.open(filename, "r") as f:
           read_data = f.read()
           temp = [line.split() for line in read_data.splitlines()]
           vocab, term_frequency = zip(*temp)
        term_frequency = [int(num) for num in term_frequency]

        vis_data = pyLDAvis.prepare(topic_term_dists=topic_term_dists, doc_topic_dists=doc_topic_dists, doc_lengths=doc_lengths, vocab=vocab, term_frequency=term_frequency)
        savename = "vw_"+senti+".html"
        pyLDAvis.save_html(vis_data, savename)

        print("Succes")
        #pyLDAvis.display(vis_data)

