import pyLDAvis

if __name__ == "__main__":
    result_dir = "C:\\Users\\kmr\\Downloads\\JST-master\\JST-master\\result\\test\\brexit\\"
    doc_lengths = []
    epoch = 56
    senti = 1
    filename = result_dir+str(epoch)+"doc_lengths.txt"
    with open(filename) as f:
       read_data = f.read()
       doc_lengths = [int(line) for line in read_data.split()]

    doc_topic_dists = []
    filename = result_dir+str(epoch)+"doc_topic_dists_theta"+str(senti)+".txt"
    with open(filename) as f:
        read_data = f.read()
        doc_topic_dists.append([line.split() for line in read_data.splitlines()])

    doc_topic_dists = [item for sublist in doc_topic_dists for item in sublist]
    doc_topic_dists = [list(map(float, doctop)) for doctop in doc_topic_dists]

    topic_term_dists = []
    filename = result_dir+str(epoch)+"topic_term_dists_phi"+str(senti)+".txt"
    with open(filename) as f:
       read_data = f.read()
       topic_term_dists.append([line.split() for line in read_data.splitlines()])

    topic_term_dists = [item for sublist in topic_term_dists for item in sublist]
    topic_term_dists = [list(map(float, doctop)) for doctop in topic_term_dists]

    vocab = []
    term_frequency = []
    filename = result_dir+str(epoch)+"vocab_term_frequency"+str(senti)+".txt"
    with open(filename) as f:
       read_data = f.read()
       temp = [line.split() for line in read_data.splitlines()]
       vocab, term_frequency = zip(*temp)
    term_frequency = [int(num) for num in term_frequency]

    vis_data = pyLDAvis.prepare(topic_term_dists=topic_term_dists, doc_topic_dists=doc_topic_dists, doc_lengths=doc_lengths, vocab=vocab, term_frequency=term_frequency)
    pyLDAvis.save_html(vis_data, "test_neg.html")
    print("Succes")
    #pyLDAvis.display(vis_data)

