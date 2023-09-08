# naivebayes for email spam detection
Spam detection model using naive bayes. Applies count vectorizer on the data then trains the model on the `spamorham.csv` dataset.

## Usage
`main.py` prepares the data, trains the model, outputs the fitnes, and saves the model to a pickle file.
Running `main.py` will retrain the model and save a new pickle file.

The accuracy of the model rarely fall below 0.97% for English inputs under 1000 characters.
