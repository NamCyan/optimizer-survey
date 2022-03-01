import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torch
import numpy as np
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
    # real_length = []
    for sentence in sentences:
        # real_length.append(len(sentence.split()))
        embedding = []
        for word in sentence.split():
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
    return np.array(embeddings)#, np.array(real_length)


def load(valid_rate, batch_size, max_length, model_name):

    savefolder = "../dat/cola_public/"

    if os.path.isfile(os.path.join(savefolder, "{}_train_{}_{}.pth".format(model_name, batch_size, max_length))):
        train_loader = torch.load(os.path.join(savefolder,"{}_train_{}_{}.pth".format(model_name, batch_size, max_length)))
        valid_loader = torch.load(os.path.join(savefolder,"{}_valid_{}_{}.pth".format(model_name, batch_size, max_length)))
        test_loader = torch.load(os.path.join(savefolder,"{}_test_{}_{}.pth".format(model_name, batch_size, max_length)))
        return train_loader, valid_loader, test_loader, 2

    trainset = pd.read_csv("../dat/cola_public/tokenized/in_domain_train.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    testset = pd.read_csv("../dat/cola_public/tokenized/in_domain_dev.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    
    train_sentences = trainset.sentence.values
    train_labels = trainset.label.values
    num_classes = len(set(train_labels))
    
    valid_length = int(valid_rate * len(trainset))
    train_length = len(trainset) - valid_length

    valid_sentences = train_sentences[:valid_length]
    valid_labels = train_labels[:valid_length]

    train_sentences = train_sentences[valid_length:]
    train_labels = train_labels[valid_length:]

    test_sentences = testset.sentence.values
    test_labels = testset.label.values

    word2emb = load_glove("../glove.6B.200d.txt")
    train_emb = get_embeddings(train_sentences, word2emb, max_length)
    valid_emb = get_embeddings(valid_sentences, word2emb, max_length)
    test_emb = get_embeddings(test_sentences, word2emb, max_length)

    def toDataLoader(emb, label, batch_size, sampler):
        ebm_ = torch.tensor(emb, dtype= torch.float)
        label_ = torch.tensor(label, dtype= torch.long)
        # real_length_ = torch.tensor(real_length, dtype= torch.long)

        TensorData = TensorDataset(ebm_, label_)
        Sampler = sampler(TensorData)
        
        Data = DataLoader(TensorData, sampler=Sampler, batch_size= batch_size)
        return Data

    train_loader = toDataLoader(train_emb, train_labels, batch_size, RandomSampler)
    valid_loader = toDataLoader(valid_emb, valid_labels, batch_size, SequentialSampler)
    test_loader = toDataLoader(test_emb, test_labels, batch_size, SequentialSampler)

    # Save process data
    torch.save(train_loader, os.path.join(savefolder, "{}_train_{}_{}.pth".format(model_name, batch_size, max_length)))
    torch.save(valid_loader, os.path.join(savefolder, "{}_valid_{}_{}.pth".format(model_name, batch_size, max_length)))
    torch.save(test_loader, os.path.join(savefolder, "{}_test_{}_{}.pth".format(model_name, batch_size, max_length)))

    return train_loader, valid_loader, test_loader, num_classes

def loadBERT(valid_rate, batch_size, max_length, tokenizer, model_name):
    savefolder = "../dat/cola_public/"

    if os.path.isfile(os.path.join(savefolder, "{}_train_{}_{}.pth".format(model_name, batch_size, max_length))):
        train_loader = torch.load(os.path.join(savefolder,"{}_train_{}_{}.pth".format(model_name, batch_size, max_length)))
        valid_loader = torch.load(os.path.join(savefolder,"{}_valid_{}_{}.pth".format(model_name, batch_size, max_length)))
        test_loader = torch.load(os.path.join(savefolder,"{}_test_{}_{}.pth".format(model_name, batch_size, max_length)))
        return train_loader, valid_loader, test_loader, 2
    
    trainset = pd.read_csv("../dat/cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    testset = pd.read_csv("../dat/cola_public/raw/in_domain_dev.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    
    train_sentences = list(trainset.sentence.values)
    train_labels = list(trainset.label.values)
    num_classes = len(set(train_labels))
    
    valid_length = int(valid_rate * len(trainset))
    train_length = len(trainset) - valid_length

    valid_sentences = train_sentences[:valid_length]
    valid_labels = train_labels[:valid_length]

    train_sentences = train_sentences[valid_length:]
    train_labels = train_labels[valid_length:]

    test_sentences = list(testset.sentence.values)
    test_labels = list(testset.label.values)

    tokenized_train = tokenizer(train_sentences, padding="max_length", truncation=True, max_length= max_length)
    tokenized_valid = tokenizer(valid_sentences, padding="max_length", truncation=True, max_length= max_length)
    tokenized_test = tokenizer(test_sentences, padding="max_length", truncation=True, max_length= max_length)

    train_token_ids, train_attention_mask, train_token_type_ids = tokenized_train["input_ids"], tokenized_train["attention_mask"], tokenized_train["token_type_ids"]
    valid_token_ids, valid_attention_mask, valid_token_type_ids = tokenized_valid["input_ids"], tokenized_valid["attention_mask"], tokenized_valid["token_type_ids"]
    test_token_ids, test_attention_mask, test_token_type_ids = tokenized_test["input_ids"], tokenized_test["attention_mask"], tokenized_test["token_type_ids"]

    def toDataLoader(input_ids, attention_mask, token_type_ids, labels, batch_size, sampler):
        input_ids_ = torch.tensor(input_ids, dtype= torch.long)
        attention_mask_ = torch.tensor(attention_mask, dtype= torch.long)
        token_type_ids_ = torch.tensor(token_type_ids, dtype= torch.long)
        labels_ = torch.tensor(labels, dtype= torch.long)
        # real_length_ = torch.tensor(real_length, dtype= torch.long)

        TensorData = TensorDataset(input_ids_, attention_mask_, token_type_ids_, labels_)
        Sampler = sampler(TensorData)
    
        Data = DataLoader(TensorData, sampler=Sampler, batch_size= batch_size)
        return Data

    train_loader = toDataLoader(train_token_ids, train_attention_mask, train_token_type_ids, train_labels, batch_size, RandomSampler)
    valid_loader = toDataLoader(valid_token_ids, valid_attention_mask, valid_token_type_ids, valid_labels, batch_size, SequentialSampler)
    test_loader = toDataLoader(test_token_ids, test_attention_mask, test_token_type_ids, test_labels, batch_size, SequentialSampler)
    
    # Save process data
    torch.save(train_loader, os.path.join(savefolder, "{}_train_{}_{}.pth".format(model_name, batch_size, max_length)))
    torch.save(valid_loader, os.path.join(savefolder, "{}_valid_{}_{}.pth".format(model_name, batch_size, max_length)))
    torch.save(test_loader, os.path.join(savefolder, "{}_test_{}_{}.pth".format(model_name, batch_size, max_length)))
    return train_loader, valid_loader, test_loader, num_classes


