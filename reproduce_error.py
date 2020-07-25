import os

if not os.path.exists('corpus_file.txt'):
    # only create file if doesn't already exist
    from random_words import RandomWords
    rw = RandomWords()
    corpus_file = open("corpus_file.txt","w")
    lengths_of_documents = open("lengths_of_documents.txt",'r').read()
    lengths_of_documents = lengths_of_documents.split()
    lengths_of_documents = [int(x) for x in lengths_of_documents if x!= '']
    #lengths_of_documents = lengths_of_documents[:100]

    for length in lengths_of_documents:
        words = rw.random_words(count=100)
        sentence = " ".join(words)* (length//100)
        corpus_file.write(sentence+"\n")

    corpus_file.close()

import gensim
from gensim.models import Doc2Vec
import logging
import time
logging.basicConfig(filename='logging_progress.log',level=logging.DEBUG)

corpus_path = "corpus_file.txt"

if os.path.isfile('model-with-vocab.doc'):
    model = Doc2Vec.load('model-with-vocab.doc')
else:
    tic = time.time()
    model = Doc2Vec(vector_size=300, min_count=100,sample=10e-5, epochs=15, workers=30)
    print ("Building vocab")
    model.build_vocab(corpus_file=corpus_path)
    print ("Size of the vocabulary: {}".format(len(list(model.wv.vocab.keys()))))
    toc = time.time()
    print ("Vocab initialization completed: {}".format(toc-tic))
    model.save('model-with-vocab.doc')

tic = time.time()

print ("training model")
model.train(corpus_file=corpus_path, total_examples=model.corpus_count,total_words=model.corpus_total_words, epochs=model.epochs)

model.save('model.doc')

toc = time.time()

print ("Training completed: {}".format(toc-tic))

