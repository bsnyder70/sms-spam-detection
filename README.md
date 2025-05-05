# Deep Learning for SMS Spam Detection: A Transformer-Based Approach 

### Abstract
In the age of reliance on constant communication between individuals and organizations, it is common for users to receive various forms of spam or scam messages, some of which can lead to stolen information and other privacy breaches. While there have been various naive attempts to remedy this issue and provide a safe messaging platform, such as keyword and area code filtering, these can easily be avoided by malicious attackers. We propose a lightweight Transformer-based classifier to detect spam in SMS messages and output flagged words, addressing the limitations of traditional filtering methods. We evaluate various tokenization strategies, loss functions, and architectural settings to optimize performance, achieving strong results even under adversarial testing.

### For more
See the full paper with model architecture details and other analysis [here](SMS_Spam_Detection_Final.pdf).

## How to Run
To train a model and infer using a given text, run:
```
python -m main.py
```


