from typing import Optional
import torch
from torch import nn
from torch import Tensor
from typing import Tuple, Union


class TransformerClassifier(nn.Module):
    """
    Custom Transformer-based Classifier designed for Spam Classification.

    The structure is: 
    1) Embedding Layer + Positional Encoding (with CLS parameter append)
    2) Transformer Encoder Layers
    3) Feedforward Neural Network (with one hidden layer)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        max_length: int,
        num_encoder_layers: int,
        class_hidden_dim: int,
    ) -> None:
        super().__init__()

        # Embedding Layers
        self.embedding_str = nn.Embedding(vocab_size, embed_dim)
        self.embedding_pos = nn.Embedding(max_length, embed_dim)

        # Transformer Encoder Layers
        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_dim,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Feedforward Neural Network Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, class_hidden_dim),
            nn.ReLU(),
            nn.Linear(class_hidden_dim, 1),
        )

    def forward(
        self,
        X: Tensor,
        get_attn_weights: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Parameters:
            X: (batch_size x seq_len) input token indices
            get_attn_weights: If True, returns attention weights from the first encoder layer

        Returns:
            logits or (logits, attention_weights)
        """

        N, T = X.shape

        # Pass input through embedding layers
        embedding_str = self.embedding_str(X)
        pos = torch.arange(T, device=X.device).expand(N, T)
        embedding_pos = self.embedding_pos(pos)
        embedding = embedding_str + embedding_pos

        # Append cls token to the embedding
        cls = self.cls_token.expand(N, -1, -1)
        encoder_input = torch.cat([cls, embedding], dim=1)

        attention_weights_l1: Optional[Tensor] = None

        # Pass encoded input through the Transformer layers
        for i, layer in enumerate(self.encoder_layers):

            # If attn weight extraction is enabled, add hook to the first layer
            if i == 0 and get_attn_weights:
                def attn_hook(module, input, output):
                    _, attn = output
                    nonlocal attention_weights_l1
                    attention_weights_l1 = attn[:, 0, :]

                handle = layer.self_attn.register_forward_hook(attn_hook)

            encoder_input = layer(encoder_input)

            if i == 0 and get_attn_weights:
                handle.remove()

        # Pass cls token output to the classifier
        cls_out = encoder_input[:, 0, :]
        logits = self.classifier(cls_out).squeeze(-1)

        if get_attn_weights and attention_weights_l1 is not None:
            return logits, attention_weights_l1
        return logits

    @staticmethod
    def from_config(**config) -> "TransformerClassifier":
        """ Return a TransformerClassifier instance with a given configuration """
        
        return TransformerClassifier(
            vocab_size=config["vocab_size"],
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            ff_dim=config["ff_dim"],
            dropout=config["dropout"],
            max_length=config["max_length"],
            num_encoder_layers=config["num_encoder_layers"],
            class_hidden_dim=config["class_hidden_dim"],
        )
