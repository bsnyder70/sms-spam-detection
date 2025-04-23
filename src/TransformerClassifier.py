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

        #self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.num_heads, dim_feedforward=self.ff_dim, dropout=self.dropout, batch_first=True)
        
        self.encoder_layers = nn.ModuleList(nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.num_heads, dim_feedforward=self.ff_dim, dropout=self.dropout, batch_first=True) 
                                            for _ in range(self.num_encoder_layers))

        #for _ in range(self.num_encoder_layers):
        #    self.encoder_layers.append(nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.num_heads, dim_feedforward=self.ff_dim, dropout=self.dropout, batch_first=True))

        #self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_encoder_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.class_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.class_hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, X, get_attn_weights=False):
        """
        Forward pass of the model

        Parameters:
            X: Input data of shape (batch_size x seq_len)
            get_attn_weights: Determines if we extract attention weights from the first layer

        Returns:
            classifier_output: Model output of shape (batch_size x 1)
        """

        N, T = X.shape

        embedding_str = self.embedding_str(X)
        pos = torch.arange(T).expand(N, T)
        embedding_pos = self.embedding_pos(pos)
        embedding = embedding_str + embedding_pos

        cls = self.cls_token.expand(N, -1, -1)

        encoder_val = torch.cat([cls, embedding], dim=1)

        attention_weights_l1 = []

        for i, layer in enumerate(self.encoder_layers):
            
            # Save the attention weights for the cls token (after the self attention forward pass)
            def attn_hook(module, input, output):
                attention_output, attention_weights = output
                attention_weights_l1 = attention_weights[:, 0, :]

            # Only add the forward hook for the first layer 
            if i == 0 and get_attn_weights:
                handle = layer.self_attn.register_forward_hook(attn_hook)

            encoder_val = layer(encoder_val)

            if i == 0 and get_attn_weights:
                handle.remove()

        cls_out = encoder_val[:, 0, :]

        classifier_output = self.classifier(cls_out).squeeze(-1)

        if get_attn_weights:
            return classifier_output, attention_weights_l1
        else:
            return classifier_output

