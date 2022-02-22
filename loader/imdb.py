import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import os
nltk.download('stopwords')
nltk.download('punkt')

def load_glove(emb_dir):
    word2emb = {}
    with open(emb_dir, "r") as f:
        for line in f.readlines():
            word = line.split()[0]
            embvec = [float(value) for value in line.split()[1:]]
            word2emb[word] = embvec
    return word2emb

def get_embeddings(sentences, word2emb, max_length): 
    english_stops = set(stopwords.words('english'))
    embeddings = []
    for sentence in sentences:
        tokenized_sent = [w.lower() for w in word_tokenize(str(sentence))]
        embedding = []
        for word in tokenized_sent:
            if word in english_stops:
                continue
            if word in word2emb:
                embedding.append(np.array(word2emb[word]))
            else:
                embedding.append(np.random.normal(0,1,200))
            if len(embedding) >= max_length: #truncation
                break
       
        if len(embedding) < max_length: #padding
            for i in range(max_length - len(embedding)):
                embedding.append(np.array([0]*200))
        embeddings.append(embedding)
    return np.array(embeddings)

def toDataLoader(emb, label, batch_size, sampler):
    ebm_ = torch.tensor(emb, dtype= torch.float)
    label_ = torch.tensor(label, dtype= torch.long)

    TensorData = TensorDataset(ebm_, label_)
    Sampler = sampler(TensorData)
    
    Data = DataLoader(TensorData, sampler=Sampler, batch_size= batch_size)
    return Data

def load(valid_rate, batch_size, max_length):
    savefolder = "../dat/imdb/"

    if os.path.isfile(os.path.join(savefolder, "train_{}_{}.pth".format(batch_size, max_length))):
        train_loader = torch.load(os.path.join(savefolder,"train_{}_{}.pth".format(batch_size, max_length)))
        valid_loader = torch.load(os.path.join(savefolder,"valid_{}_{}.pth".format(batch_size, max_length)))
        test_loader = torch.load(os.path.join(savefolder,"test_{}_{}.pth".format(batch_size, max_length)))
        return train_loader, valid_loader, test_loader, 2
    
    fulldata = pd.read_csv("../dat/imdb/imdb.csv")
    data_sentences = fulldata.review.values
    data_labels = fulldata.sentiment.values

    num_classes = len(set(data_labels))
    label_dict = {}
    for i, label in enumerate(set(data_labels)):
        label_dict[label.lower()] = i
    label_ids = []
    for label in data_labels:
        label_ids.append(label_dict[label])

    train_sentences, test_sentences, train_labels, test_labels = train_test_split(data_sentences, label_ids, train_size= 0.8, test_size= 0.2, random_state= 0)  
    train_sentences, valid_sentences, train_labels, valid_labels = train_test_split(train_sentences, train_labels, train_size= 1 - valid_rate, test_size= valid_rate, random_state= 0)

    print("Loading embedding ...")
    word2emb = load_glove("../glove.6B.200d.txt")
    print("Vocabulary loaded")
    train_emb = get_embeddings(train_sentences, word2emb, max_length)
    print("Train set loaded")
    valid_emb = get_embeddings(valid_sentences, word2emb, max_length)
    print("Valid set loaded")
    test_emb = get_embeddings(test_sentences, word2emb, max_length)
    print("Test set loaded")
    print("Done")
    train_loader = toDataLoader(train_emb, train_labels, batch_size, RandomSampler)
    valid_loader = toDataLoader(valid_emb, valid_labels, batch_size, SequentialSampler)
    test_loader = toDataLoader(test_emb, test_labels, batch_size, SequentialSampler)

    # Save process data
    torch.save(train_loader, os.path.join(savefolder, "train_{}_{}.pth".format(batch_size, max_length)))
    torch.save(valid_loader, os.path.join(savefolder, "valid_{}_{}.pth".format(batch_size, max_length)))
    torch.save(test_loader, os.path.join(savefolder, "test_{}_{}.pth".format(batch_size, max_length)))
    return train_loader, valid_loader, test_loader, num_classes




