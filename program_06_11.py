import pandas as pd
#import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#from textblob import TextBlob 
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import re

#print("Tensorflow Version",tf.__version__)
df = pd.read_csv('tweets.csv',encoding='latin-1')
df.columns = ['sentiment', 'id', 'date', 'query', 'user_id', 'text']

df = df.drop(['id', 'date', 'query', 'user_id'], axis=1)
df.loc[df['sentiment'] == 4, 'sentiment'] = 1




stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')

clean_chars = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
def preprocess(text, stem=False):
    
    text = re.sub(clean_chars, ' ', str(text).lower())
    text = text.strip()

    tokens = []
    for word in text.split():
        if word not in stop_words:
            if stem:
                tokens.append(stemmer.stem(word))
            else:
                tokens.append(word)
    return " ".join(tokens)
df.text = df.text.apply(lambda x: preprocess(x))
df.text = df.text.apply(preprocess)

positive_tweets = df[df.sentiment == 1]
negative_tweets = df[df.sentiment == 0]


from collections import Counter 
#most frequent positive words

pos_words = " ".join(positive_tweets.text)
split_it = pos_words.split() 

positive_word_freq = Counter(split_it)

most_occur = positive_word_freq.most_common(15) 
words, frequencies = zip(*most_occur)
plt.figure(figsize=(10, 6))
plt.bar(words, frequencies)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 15 Most Occurring Words in positive Tweets')



neg_words = " ".join(negative_tweets.text)
split_it2 = neg_words.split() 

negative_word_freq = Counter(split_it2)
most_occur2 = negative_word_freq.most_common(15) 
words2, frequencies2 = zip(*most_occur2)
plt.figure(figsize=(10, 6))
plt.bar(words2, frequencies2)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 15 Most Occurring Words in negative Tweets')
 
 

tfidf_vectorizer = TfidfVectorizer(max_features=1000000)
X = tfidf_vectorizer.fit_transform(df['text'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')

print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()