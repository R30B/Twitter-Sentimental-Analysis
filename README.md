# Twitter-Sentimental-Analysis
By analysing tweets it detects if tweet is related to hate speech or not

I used pandas for extract csv files and for data manipulation,Sklearn for getting machine learning modules.

Procedure:

1. Initially I removed all the special characters(noise) in the tweets that has no use in understanding tweets.

2. I used Spacy library for converting the words into its root word called lemmatization

3. Removed all the stop words

4. Splitting the training set data into train set and validation set to improve the model.

5. Used the TF-IDF Vectorizer in train set and validation set (For transforming words into vectors) .

6. Used SGDClassifier (SGD stands for Stochastic Gradient Descent) with hinge loss (which is equivalent to Linear Support Vector Machine(SVM) but much faster) to train the model.

7. Applied the same model to predict the labels or classes for test set and store it in csv.
