import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

df = pd.read_csv('spamorham.csv')
# training set is a little unbalanced
# print(df.groupby('Category').describe())     to see ratio

df['spam'] = df['Category'].apply(lambda x: 1 if x =='spam' else 0)
df = df.drop('Category', axis=1)

x_train, x_test, y_train, y_test = train_test_split(df.Message,df.spam, test_size=0.2)

clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())])
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))

with open('spamdetector.pickle', 'wb') as f:
    pickle.dump(clf, f)









