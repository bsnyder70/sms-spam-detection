import torch
import numpy as np
import pandas as pd
import kagglehub
from torch.utils.data import Dataset
import re
from collections import Counter
import pickle

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
        """
        Initialize the Vocabulary object. We insert the special tokens
        into our storage initially before we fill in the rest of the words.
        """

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
        Given a list of tokenized strings, build the Vocabulary, mapping each string to an index.
        """

        # Fill the counter with all the input words (in all examples)
        cnt = Counter()
        for example in examples:
            cnt.update(example)

        # Update the storage only with words that have more than one instance. 
        idx = len(self.stoi)
        for word, count in cnt.items():
            if count >= self.min_freq and word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
       
    def get_itos(self):
        """ Return the index-to-string mapping. """
        return self.itos

def download_data():
    """ Downloads the Spam Collection dataset using the Kaggle API """

    # Download latest version
    path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset", )

    return path

def build_data(save_vocab_to_file=False):
    """
    Download, process, and format the dataset. The major steps include:
    1) Download dataset and load into a Pandas DataFrame.
    2) Pre-process and tokenize the text.
    3) Using the tokens, build the vocabulary.
    4) Convert the examples and labels into Tensors and load into a custom Torch DataSet.

    Parameters:
        None

    Returns: 
       SpamDataset: Custom Torch Dataset used to store all the examples
       vocab_size: The number of unique tokens stored in the vocabulary

    """

    # Download data from Kaggle
    path = download_data()

    # Read the downloaded data and format the dataframe as necessary
    df = pd.read_csv(f"{path}/spam.csv", encoding='latin-1')[['v1', 'v2']]
    df.rename(columns={'v1': 'labels', 'v2': 'examples'}, inplace=True)

    # Map the labels to integer values for binary classification
    df['labels'] = df['labels'].map({'ham': 0, 'spam': 1})

    # Need to go through all the vocabulary, map to integers, calculate the size of vocab, add other tokens (cls, etc)

    # Clean and tokenize the example text.
    df['examples'] = df['examples'].apply(preprocess_text)

    # Convert the labels to tensors
    labels = torch.tensor(df['labels'].values, dtype=torch.float)

    examples = df['examples'].values

    # Build the Vocabulary
    vocab = Vocabulary(min_freq=2)
    vocab.build_vocab(examples)
    vocab_size = len(vocab)

    # Save the Vocabulary to a file
    if save_vocab_to_file:
        filename = 'cache/vocab.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(vocab, file)

    # Map the words to ids
    examples_tensor = torch.tensor([[vocab[word] for word in example] for example in examples])

    spam_dataset = SpamDataset(labels, examples_tensor)

    return spam_dataset, vocab_size

def preprocess_text(text):
    """
    Preprocess and tokenize the example text. 

    Parameters:
        text: Input text to be processed

    Returns:
        words: List of tokenized, preprocessed words.
    """
    MAX_SEQ_LEN = 180

    # We will start with just removing any non alphabetic characters and making everything lowercase
    # should change in future though, extract numbers/emails/etc

    # Change commas to spaces to preserve separation
    text = re.sub(r'[,]',' ', text)

    # Remove non alphabetic/space characters.
    text = re.sub(r'[^a-zA-Z ]', '', text)

    # Change all letters to lowercase for uniformity.
    text = text.lower()

    # Split on whitespace (and commas due to the above regex) to create tokens
    words = text.split(" ")
    
    # Insert cls and end of sentence token, then pad up to the max sequence length.
    words.insert(0, "<cls>")
    words.append("<eos>")

    for i in range(len(words), MAX_SEQ_LEN):
        words.append("<pad>")
    
    return words

def get_top_k_words(tokens, attention_weights, vocabulary, k=3):
    """
    Return the top k attended to tokens (as given by attention_weights).
    """

    attention_weights_indexed = [(attention_weights[i],i) for i in range(len(attention_weights))]
    attention_weights_indexed.sort(reverse=True)

    top_k_attention_weights = attention_weights_indexed[:k]
    top_k_tokens_idx = [tokens[i] for _, i in top_k_attention_weights]
    top_k_tokens = [vocabulary.itos(idx) for idx in top_k_tokens_idx]

    return top_k_tokens

def get_input_from_text(text, vocabulary):
    """
    Clean and tokenize text in preparation for model inference.
    """

    words = preprocess_text(text)
    words_idx = torch.tensor([vocabulary[word] for word in words_idx]).reshape(1, -1)

    return words_idx

build_data()