# Ritwick Bisht
# Section - A
# University Roll No - 2013458
import numpy as np
#for handle csv files
import pandas as pd
# for using bar plots and confusion matrix
from matplotlib import pyplot as plt
import seaborn as sns
# for using tfidf vectors
from sklearn.feature_extraction.text import TfidfVectorizer
# for using Support Vector Machine Model
from sklearn.linear_model import SGDClassifier
# for f1_score and accuracy
from sklearn.metrics import f1_score,accuracy_score
#train test splitting
from sklearn.model_selection import train_test_split
# for using Natural Processing technique called Lemmatization
import spacy
plt.style.use('ggplot')
pd.set_option("display.max_colwidth",None)
X_train = pd.read_csv('train_data.csv') # to read the training set of csv files
X_test = pd.read_csv('test_data.csv') #to read the test set of csv files
nlp = spacy.load('en_core_web_sm')


def clean_it(tweet, model):
    # to replace the @user in twitter
    tweet = tweet.str.replace('@[\w]*', '', regex=True)
    # to replace any special character in twitter
    tweet = tweet.str.replace('[^a-zA-Z]', ' ', regex=True)
    # lemmatizing the sentences
    with model.select_pipes(enable=['tagger', 'lemmatizer', 'attribute_ruler', 'tok2vec']):
        token = pd.Series(model.pipe(tweet))

    filtered_tokens = pd.Series(
        [[x.lemma_ for x in tok if x.is_stop == False] for tok in token])  # to remove stop words

    for i in range(len(filtered_tokens)):
        filtered_tokens[i] = ' '.join(filtered_tokens[i])

    return filtered_tokens

X_train.tweet = clean_it(X_train.tweet,nlp)
X_test.tweet = clean_it(X_test.tweet,nlp)
y=X_train.pop('label')
#to split it into training and validation set
Xtrain,Xvalid,ytrain,yvalid=train_test_split(X_train,y,random_state=0,test_size=0.3)
# using Tfidf Vectorizizer
tfidf = TfidfVectorizer(ngram_range=(1,3),max_features=800000)
# to convert it into tfidf vectors
tfidf_train_set = tfidf.fit_transform(Xtrain.tweet)
# tranforming it into the validation set
tfidf_valid_set = tfidf.transform(Xvalid.tweet)
svm = SGDClassifier(max_iter=1000,alpha =1e-5,loss='hinge',penalty='l2')  # using Support Vector Machine

svm.fit(tfidf_train_set,ytrain)  # fitting the model or training the model
ypred=svm.predict(tfidf_valid_set) # predicting the data in validation set

print(accuracy_score(yvalid,ypred)) # accuracy of the validation set
print(f1_score(yvalid,ypred)) # f1 score of the validation set

tfidf_train = tfidf.fit_transform(X_train.tweet) # now converting the all dataset into tfidf vectorizer
svm.fit(tfidf_train,y) # training the model
tfidf_test = tfidf.transform(X_test.tweet) # transfroming the the tweets into tfidf vectors
y_test_pred = svm.predict(tfidf_test) # predicting the data in the test set(which is unlabelled)

X_test['label'] = y_test_pred

submission_csv = pd.DataFrame({'id':X_test.id,'label':y_test_pred})
# to save the data into csv files
submission_csv.to_csv('submission_svm.csv',index = False)