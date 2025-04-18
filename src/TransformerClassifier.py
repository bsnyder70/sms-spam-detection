import torch
from torch import nn

class TransformerClassifier(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, dropout, max_length, num_encoder_layers, class_hidden_dim):
        """
        Initializes our Transformer Classifier.

        The basic structure is:
        Input -> Embedding + Positional Embedding -> Transformer Encoder -> FNN Classifier
        """

        super(TransformerClassifier, self).__init__()

        self.vocab_size = 3963
        self.embed_dim = 128
        self.num_heads = 4
        self.ff_dim = 128
        self.dropout = 0.3
        self.max_length = 180
        self.num_encoder_layers = 2
        self.class_hidden_dim = 64

        self.embedding_str = nn.Embedding(self.vocab_size, self.embed_dim)
        self.embedding_pos = nn.Embedding(self.max_length, self.embed_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.num_heads, dim_feedforward=self.ff_dim, dropout=self.dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_encoder_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.class_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.class_hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        """
        Forward pass of the model

        Parameters:
            X: Input data of shape (batch_size x seq_len)

        Returns:
            classifier_output: Model output of shape (batch_size x 1)
        """

        N, T = X.shape

        embedding_str = self.embedding_str(X)
        pos = torch.arange(T).expand(N, T)
        embedding_pos = self.embedding_pos(pos)
        embedding = embedding_str + embedding_pos

        cls = self.cls_token.expand(N, -1, -1)

        input = torch.cat([cls, embedding], dim=1)

        encoder_output = self.encoder(input)
        cls_out = encoder_output[:, 0, :]

        classifier_output = self.classifier(cls_out).squeeze(-1)

        return classifier_output

