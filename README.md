# dl-final
 
Repoistory for our final project for Georgia Tech OMSCS Deep Learning.

Once we are done, we should clean up this readme so we can use it on our portfolios

## TODO / Look at
- Experiment with different types of tokenization / embedding. Right now we just remove all non alphabetic characters, but we may want to experiment with char level embeddings + keeping the numbers/selected punctuation. This would require some change to the Vocabulary class
- Run analysis on the input data set. How many examples are there for each class? Do the lengths of the classes differ? Most common words in the Vocabulary?
- Build lots of tuning / data graphs
    - Make functions for visual confusion matrix, precision/recall
    - Training graphs, see train/val loss/accuracy curves
- Make a more formal config
    - With this, ways to easily tune / grid search for best parameters
- Make a way to save the model so we don't have to constantly retrain.
- Add Top K tokens processing. On evaluating the model, find the top K tokens that the transformer attended to to lead to that decision, then convert back to strings and return.
    - Future extensions include building heat map of most important words globally, most impotant words for spam, most import words for non spam
- Shuffle the data to account for the imbalance between classes.
