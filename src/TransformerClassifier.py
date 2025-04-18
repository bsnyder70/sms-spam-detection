import torch
from torch import nn

class TransformerClassifier(nn.Module):

    def __init__(self):
        """
        
        :param X: xxx

        :returns: XXX
        """

        super(TransformerClassifier, self).__init__()

        self.vocab_size = 5
        self.embed_dim = 5
        self.num_heads = 5
        self.ff_dim = 5
        self.dropout = 0.5
        self.max_length = 10
        self.num_layers = 5
        self.class_input_dim = 5

        self.embedding_str = nn.Embedding(self.vocab_size, self.embed_dim)
        self.embedding_pos = nn.Embedding(self.max_length, self.embed_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.num_heads, dim_feedforward=self.ff_dim, dropout=self.dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        """
        
        :param X: xxx

        :returns: XXX
        """

        N, seq_len = X.shape

        embedding_str = self.embedding_str(X)
        embedding_pos = self.embedding_pos(X)
        embedding = embedding_str + embedding_pos

        cls = self.cls_token.expand(N, -1, -1)

        input = torch.cat([cls, embedding], dim=1)

        encoder_output = self.encoder(input)
        cls_out = encoder_output[:, 0, :]

        classifier_output = self.classifer(cls_out).squeeze(-1)

        return classifier_output

