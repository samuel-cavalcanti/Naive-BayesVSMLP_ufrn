import keras
import numpy as np
import matplotlib.pyplot as plt

def vetorize_sequences(sequences,dimension=10000):
    results = np.zeros( (len(sequences),dimension))

    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1 
    
    return results


def plotTraining(label_train,label_val,title,train_data,val_data,plot=False):
    plt.figure()
    plt.plot(epochs,train_data,"bo",label=label_train)
    plt.plot(epochs,val_data,"b",label=label_val)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show(block=False)

    if plot:
        plt.show()
    





(train_data,train_labels),(test_data,test_labels)= keras.datasets.imdb.load_data(num_words=10000)
word_index = keras.datasets.imdb.get_word_index()

reverse_word_index = dict([(value,key) for (key,value) in word_index.items() ] )

decoded_review = " ".join([reverse_word_index.get( i-3,"?") for i in train_data[0]])

x_train = vetorize_sequences(train_data)
x_test = vetorize_sequences(test_data)

y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")


input_layer = keras.layers.Input(shape=(10000,) )
first_hidden_layer  = keras.layers.Dense(16,activation="relu") (input_layer)
second_hidden_layer = keras.layers.Dense(16,activation="relu") (first_hidden_layer)

output_layer = keras.layers.Dense(1,activation="sigmoid") (second_hidden_layer)

network = keras.models.Model(inputs=input_layer,outputs=output_layer)

network.compile(optimizer=keras.optimizers.RMSprop(lr=0.001), loss=keras.losses.binary_crossentropy,metrics=[keras.metrics.binary_accuracy] )


x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]


history = network.fit(partial_x_train,partial_y_train,epochs=4,batch_size=512,validation_data=(x_val,y_val) )

history_dict = history.history

acc = history.history["binary_accuracy"]
val_acc = history.history["val_binary_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1,len(acc) +1 )

results = network.evaluate(x_test,y_test)
print(results)


plotTraining("Training loss","Validation loss","Training and validation loss",loss,val_loss)
plotTraining("Training acc","Validation acc","Training and validation accuracy",acc,val_acc,plot=True)





