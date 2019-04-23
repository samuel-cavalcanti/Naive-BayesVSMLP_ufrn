from keras.datasets import imdb
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def vetorize_sequences(sequences,dimension=10000):
    results = np.zeros( (len(sequences),dimension))

    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1 
    
    return results



(train_data,train_labels),(test_data,test_labels)= imdb.load_data(num_words=10000)

x_train = vetorize_sequences(train_data)
x_test = vetorize_sequences(test_data)


model = MultinomialNB()
model.fit(x_train,train_labels)
preds = model.predict(x_test)

score = roc_auc_score(test_labels,preds)



