import pickle
import random

import numpy as np

from new_train import train
from tuning import tuning
from utils.loss import (
    FocalLoss,
    binary_cross_entropy_loss,
    get_loss_wrapper,
    get_pos_weight,
    weighted_bce_with_logits_loss,
)
import os
import torch
from torch import nn
import pandas as pd
from new_train import train
from tuning import tuning
from utils.loss import FocalLoss, binary_cross_entropy_loss, get_loss_wrapper, get_pos_weight, \
    weighted_bce_with_logits_loss
from model.config import default_config
from model.TransformerClassifier import TransformerClassifier
from train import generate_train_test, evaluate
from sklearn.metrics import confusion_matrix, classification_report
from data_process import (
    build_data,
    preprocess_text,
    preprocess_text_minimal,
    preprocess_no_special_tokens,
    preprocess_raw,
    preprocess_stemmed, get_input_from_text, preprocess_wordpiece,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    seed_everything(42)

    # Map a name to each preprocessing strategy
    strategies = {
        'default': preprocess_text,
        'minimal': preprocess_text_minimal,
        'no_special': preprocess_no_special_tokens,
        'raw': preprocess_raw,
        'stemmed': preprocess_stemmed,
        'wordpiece': preprocess_wordpiece
    }

    for name, proc_fn in strategies.items():
        print(f"\n\n=== Training with '{name}' tokenization ===")

        vocab_file_path = f"cache/{name}.vocab.pkl"
        # Rebuild dataset & vocabulary with this strategy
        dataset, vocab_size = build_data(preprocess_fn=proc_fn, save_vocab_to_file=True, vocab_path=vocab_file_path)

        config = default_config.copy()
        config["vocab_size"] = vocab_size

        train_loader, valid_loader, test_loader = generate_train_test(
            dataset=dataset,
            batch_size=config["batch_size"]
        )

        model = TransformerClassifier.from_config(**config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        criterion = FocalLoss(alpha=0.5, gamma=1.0)
        loss_used = 'focal'

        # criterion = get_loss_wrapper(binary_cross_entropy_loss, apply_sigmoid=True)
        # loss_used = 'BCE'

        # Save each model under a distinct name
        save_path = f"outputs/{name}_{loss_used}_model.pth"
        train(
            num_epochs=config["num_epochs"],
            train_loader=train_loader,
            valid_loader=valid_loader,
            model=model,
            optimizer=optimizer,
            device=device,
            loss_fn=criterion,
            save_path=save_path,
        )

        # Evaluate on test set
        test_loss, test_acc, preds, labels = evaluate(
            model,
            test_loader,
            criterion,
            device=device
        )
        print(f"[{name}] Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        cm = confusion_matrix(labels, preds)
        print(cm)
        print(classification_report(labels, preds, target_names=["Ham", "Spam"]))

        # # Tune the model

        # 1. grab config from config.py
        #model_config = default_config.copy()

        # binary cross entropy loss
        #tuning(model_config, dataset, model, criterion)

        # # weighted binary cross entropy loss
        # # Compute pos_weight for spam vs ham
        # pos_weight = get_pos_weight(dataset)

        # # Create loss function that handles imbalance
        # loss_fn = weighted_bce_with_logits_loss(pos_weight)

        # # Tuning
        # tuning(
        #     model_config,
        #     dataset,
        #     TransformerClassifier,
        #     loss_fn,
        # )

        # # Get base pos_weight
        # base_pos_weight = get_pos_weight(dataset)
        # results = []

        # # Try different scaled versions
        # for multiplier in [0.5, 1.0, 1.5, 2.0]:
        #     scaled_weight = base_pos_weight * multiplier
        #     print(
        #         f"\nRunning tuning with pos_weight={scaled_weight:.2f} (multiplier={multiplier})"
        #     )

        #     # Define loss function
        #     loss_fn = weighted_bce_with_logits_loss(scaled_weight)

        #     # Optional: label for plots/logs
        #     model_config["loss_label"] = f"pos_weight={scaled_weight:.2f}"

        #     best = tuning(
        #         config=model_config,
        #         dataset=dataset,
        #         model_cls=TransformerClassifier.from_config(model_config),
        #         loss_fn=loss_fn,
        #         sweep_label=f"pos_weight_{scaled_weight:.2f}",
        #     )

        #     best["pos_weight"] = scaled_weight

        #     results.append(best)
        # # Save all best configs
        # pd.DataFrame(results).sort_values("f1", ascending=False).to_csv(
        #     "outputs/pos_weight_sweep_summary.csv", index=False
        # )

        # # Focal loss tuning
        # alphas = [0.25, 0.5, 1.0]
        # gammas = [1.0, 2.0, 3.0]

        # results = []

        # for alpha in alphas:
        #     for gamma in gammas:
        #         label = f"focal_a{alpha}_g{gamma}"
        #         print(f"\n Tuning for alpha={alpha}, gamma={gamma}")
        #         loss_fn = FocalLoss(alpha=alpha, gamma=gamma)

        #         best = tuning(
        #             config=model_config,
        #             dataset=dataset,
        #             model_cls=TransformerClassifier,
        #             loss_fn=loss_fn,
        #             sweep_label=label,
        #         )

        #         best["alpha"] = alpha
        #         best["gamma"] = gamma
        #         results.append(best)

        # # Save all best configs
        # pd.DataFrame(results).sort_values("f1", ascending=False).to_csv(
        #     "outputs/focal_sweep_summary.csv", index=False
        # )



def run_model(text=None):
    """ Run inference on a saved model and output the prediction + Top K words. """

    config = default_config.copy()

    text = "Thanks for your subscription to Ringtone UK your mobile will be charged �5/month Please confirm by"

    # Load the Vocabulary object from the file
    filename = 'cache/vocab.pkl'
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            vocabulary = pickle.load(file)
    else:
        print("Missing Vocabulary file. Try rebuilding the dataset.")
        return

    # Tokenize the input text
    input = get_input_from_text(text, vocabulary)

    # Load the existing model
    model = TransformerClassifier.from_config(**config)
    # model_state_path = 'outputs/bce_model_tst.pth'
    model_state_path = 'outputs/focal_model.pth'
    model_state = torch.load(model_state_path)
    model.load_state_dict(model_state)
    model.eval()

    #logits, attention_weights = model.forward(input, get_attn_weights=True)
    #vals = torch.sigmoid(logits)
    #
    # # Calculate probabilities and Top K
    # if vals[0] < 0.5:
    #     likelihood = (0.5 - vals[0]) * 200
    #     typ = "Not Spam"
    # else:
    #     likelihood = (vals[0] - 0.5) * 200
    #     typ = "Spam"
    #
    # print(f"The message is classified as {typ} with a probability of {likelihood}%")
    #
    # top_k_words = get_top_k_words(input.squeeze(0).tolist(), attention_weights.squeeze(0).tolist()[1:], vocabulary)
    #
    # print(f"The top words that were used to determines this are: {','.join(top_k_words)}")

main()
# run_model()
