default_config = {
    # Training parameters
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 1e-4,
    # Model architecture parameters
    "vocab_size": 3963,
    "embed_dim": 128,
    "num_heads": 4,
    "ff_dim": 128,
    "dropout": 0.0,
    "max_length": 180,
    "num_encoder_layers": 2,
    "class_hidden_dim": 64,
}


focal_best_config = {
    # Training parameters
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 1e-4,
    # Model architecture parameters
    "vocab_size": 3963,
    "embed_dim": 256,
    "num_heads": 4,
    "ff_dim": 128,
    "dropout": 0.0,
    "max_length": 180,
    "num_encoder_layers": 2,
    "class_hidden_dim": 64,
    # Loss-specific parameters
    # "alpha": 0.25,
    # "gamma": 3.0,
}

weighted_bce_best_config = {
    # Training parameters
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 1e-4,
    # Model architecture parameters
    "vocab_size": 3963,
    "embed_dim": 64,
    "num_heads": 4,
    "ff_dim": 128,
    "dropout": 0.0,
    "max_length": 180,
    "num_encoder_layers": 2,
    "class_hidden_dim": 64,
    # Loss-specific parameters
    "loss_label": "pos_weight=3.23",
    "pos_weight": 3.23,
}


bce_best_config = {
    # Training parameters
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 1e-4,
    # Model architecture parameters
    "vocab_size": 3963,
    "embed_dim": 128,
    "num_heads": 4,
    "ff_dim": 128,
    "dropout": 0.0,
    "max_length": 180,
    "num_encoder_layers": 2,
    "class_hidden_dim": 64,
}
