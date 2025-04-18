import numpy as np
import pandas as pd
import kagglehub
from torch.utils.data import Dataset
import re

class SpamDataset(Dataset):

    def __init__(self, labels, examples):
        self.labels = labels
        self.examples = examples
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.examples[index], self.labels[index]
    
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

    print(df.head(5))

    # clean examples and tokenize as per assignment 3
    df['examples'] = df['examples'].apply(preprocess_text)

    labels = df['labels'].values
    examples = df['examples'].values

    print(df.head(5))

    spam_dataset = SpamDataset(labels, examples)

    return spam_dataset

def preprocess_text(text):
    """
    :param text: input text

    :return list of preprocessed words
    """
    # we will start with just removing any non alphabetic characters and making everything lowercase
    # should change in future though, extract numbers/emails/etc

    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()

    words = text.split(" ")

    return words


build_data()