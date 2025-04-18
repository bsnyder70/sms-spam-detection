import torch
import numpy as np
import pandas as pd
import kagglehub
from torch.utils.data import Dataset
import re
from collections import Counter

class SpamDataset(Dataset):

    def __init__(self, labels, examples):
        self.labels = labels
        self.examples = examples
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.examples[index], self.labels[index]

class Vocabulary():

    def __init__(self, min_freq=2, special_tokens=None):
        if special_tokens is None:
            special_tokens = ['<cls>','<eos>','<pad>','<unk>']

        self.stoi = {token: idx for idx, token in enumerate(special_tokens)}
        self.itos = {idx: token for idx, token in enumerate(special_tokens)} 
        self.min_freq = min_freq

    def __len__(self):
        return len(self.stoi)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi['<unk>'])

    def __call__(self, token):
        return self.stoi.get(token, self.stoi['<unk>'])
    
    def build_vocab(self, examples):
        """
        Given a list of tokenized strings, build the vocabulary mapping each string to an index.
        """
        cnt = Counter()
        for example in examples:
            cnt.update(example)

        idx = len(self.stoi)
        for word, count in cnt.items():
            if count >= self.min_freq and word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
       
    def get_itos(self):
        """Return the index-to-string mapping."""
        return self.itos
    
def download_data():
    # Download latest version
    path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset", )

    return path

def build_data():

    # Download data from Kaggle
    path = download_data()

    # Read the downloaded data and format the dataframe as necessary
    df = pd.read_csv(f"{path}/spam.csv", encoding='latin-1')[['v1', 'v2']]
    df.rename(columns={'v1': 'labels', 'v2': 'examples'}, inplace=True)
    df['labels'] = df['labels'].map({'ham': 0, 'spam': 1})

    # Need to go through all the vocabulary, map to integers, calculate the size of vocab, add other tokens (cls, etc)

    # clean examples and tokenize as per assignment 3
    df['examples'] = df['examples'].apply(preprocess_text)

    # Convert the labels to tensors
    labels = torch.tensor(df['labels'].values, dtype=torch.float)

    examples = df['examples'].values

    vocab = Vocabulary(min_freq=2)
    vocab.build_vocab(examples)

    examples_tensor = torch.tensor([[vocab[word] for word in example] for example in examples])

    # so vocab is of size 3965 as of now
    #print(f'vocab size: {len(vocab)}')
    #print(f'{max(vocab.get_itos().keys())}')

    spam_dataset = SpamDataset(labels, examples_tensor)

    return spam_dataset

def preprocess_text(text):
    """
    :param text: input text

    :return list of preprocessed words
    """
    # we will start with just removing any non alphabetic characters and making everything lowercase
    # should change in future though, extract numbers/emails/etc

    # Change commas to spaces to preserve separation
    text = re.sub(r'[,]',' ', text)

    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()

    words = text.split(" ")
    
    # VERY crude tokenization and padding
    MAX_SEQ_LEN = 180
    words.insert(0, "<cls>")
    words.append("<eos>")

    for i in range(len(words), MAX_SEQ_LEN):
        words.append("<pad>")
    
    return words




build_data()