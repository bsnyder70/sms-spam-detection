import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt
from model.TransformerClassifier import TransformerClassifier
from data_process import preprocess_text, preprocess_wordpiece
from sklearn.metrics import classification_report


def load_vocab(vocab_path: str):
    """
    Load a Vocabulary object from a .pkl file.
    """
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = load_vocab("cache/wordpiece.vocab.pkl")


def text_to_tensor(text):
    tokens = preprocess_wordpiece(text)           # ← use WordPiece
    token_ids = torch.tensor([vocab[w] for w in tokens])
    return token_ids.unsqueeze(0).to(device)


model = TransformerClassifier(
    vocab_size=len(vocab), embed_dim=128, num_heads=4,
    ff_dim=128, dropout=0.0, max_length=180,
    num_encoder_layers=2, class_hidden_dim=64
)
model.load_state_dict(torch.load("outputs/wordpiece_focal_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Test sets
originals = [
    "Win a free iPhone now!",
    "You have won $1000. Click the link to claim.",
    "Congratulations! You're selected for a prize.",
    "Free entry in our weekly draw. Hurry up!",
    "Claim your reward now!"
]

adversarial_spam = [
    "W1n a fr33 iPh0ne n0w!",
    "You have won $1000. Cl1ck the l1nk to cla1m.",
    "Congr@tulations! You are selected 4 a pr1ze.",
    "Free entry in our w e e k l y draw! Hurry up!!",
    "Claim y0ur re-ward n0w!!!"
]

extreme_spam = [
    "!!! FrEEEEE iPHonE. g3t n0w www.claimwinnings[dot]com !!!",
    "WIN big$$$$ CLICK N0W hxxp[:]//f4ke-link.biz",
    "CONGRATZZ u w0n b1g!!1! Check ur email.",
    "Weekly Draw: WIN! Claim at ur link!!! FREE.",
    "HURRY!!! $$$$ Pr1zE Awaits y0u. T0day 0nly."
]

original_ham = [
    "Hey, are you coming to dinner tonight?",
    "Don't forget to bring the documents tomorrow.",
    "Happy birthday! Wishing you the best!",
    "Let's catch up over coffee sometime.",
    "Meeting moved to 3pm. Let me know if that works."
]

adversarial_ham = [
    "Dinner Reminder!!! RSVP NOW for Tonight!",
    "Important Docs Due Tomorrow!!! Confirm!!!",
    "Birthday Alert! 🎉 Confirm Your RSVP!",
    "Coffee Meeting Invitation!!! Reply ASAP!",
    "Schedule Change!!! Confirm 3PM Meeting Now!"
]


def predict(text):
    tensor = text_to_tensor(text).to(device)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).item()
    return 1 if prob > 0.5 else 0, prob


def run_evaluation(title, originals, adversarials, label, confidences_dict, key_name):
    print(f"\n=== {title} ===")
    print(f"{'Original':<55} {'Adv/Extreme Version':<55} | Orig → Adv | Spam? | Conf")
    print("-" * 130)

    all_preds, all_labels = [], []
    confidences = []

    for orig, adv in zip(originals, adversarials):
        orig_pred, orig_conf = predict(orig)
        adv_pred, adv_conf = predict(adv)

        flipped = "YES" if orig_pred != adv_pred else "NO"
        all_preds.append(adv_pred)
        all_labels.append(label)
        confidences.append(adv_conf)

        print(
            f"{orig[:52]:<55} {adv[:52]:<55} | {orig_pred} → {adv_pred}  |  {'SPAM' if adv_pred else 'HAM'} | {adv_conf:.2f}")

    confidences_dict[key_name] = confidences

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, labels=[0, 1], target_names=["Ham", "Spam"], zero_division=0))


def plot_confidences(confidences_dict):
    labels = ["Example 1", "Example 2", "Example 3", "Example 4", "Example 5"]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width, confidences_dict['Light Adversarial Spam'], width, label='Light Adversarial Spam')
    bars2 = ax.bar(x, confidences_dict['Extreme Adversarial Spam'], width, label='Extreme Adversarial Spam')
    bars3 = ax.bar(x + width, confidences_dict['Adversarial Ham'], width, label='Adversarial Ham')

    ax.set_ylabel('Spam Confidence Score')
    ax.set_title('Model Confidence: Spam vs Adversarial Ham Examples')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend()

    # Add confidence values on top of bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    confidences_dict = {}
    run_evaluation("Original Spam → Light Adversarial Spam", originals, adversarial_spam, label=1,
                   confidences_dict=confidences_dict, key_name="Light Adversarial Spam")
    run_evaluation("Original Spam → Extreme Perturbation Spam", originals, extreme_spam, label=1,
                   confidences_dict=confidences_dict, key_name="Extreme Adversarial Spam")
    run_evaluation("Original Ham → Adversarial Spammy-Looking Ham", original_ham, adversarial_ham, label=0,
                   confidences_dict=confidences_dict, key_name="Adversarial Ham")

    plot_confidences(confidences_dict)
