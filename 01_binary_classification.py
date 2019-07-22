import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import imdb
from keras import models, layers

# Get the training data/ labels

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# >>> train_data[0]
# [1, 14, 22, 16, ... 178, 32]

# >>> train_data.shape
# (25000,)

# >>> train_labels[0]
# 1

# >>> train_labels.shape
# (25000,)

# >>> max([max(sequence) for sequence in train_data])
# 9999


def decoded_review(data, index):
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in data[index]])


# >>> decoded_review(train_data, 0)
# "? this film was just brilliant casting location scenery story direction everyone's ..."


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


# Vectorize the training/ test data

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Vectorize the training/ test labels

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# Create the model

model = models.Sequential()

# Add the layers

# model.add(layers.Dense(32, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(32, activation='relu')) # higher layer size --> minor change in accuracy
# model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(64, activation='relu')) # higher layer size --> minor change in accuracy
# model.add(layers.Dense(8, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(8, activation='relu')) # lower layer size --> minor change in accuracy
# model.add(layers.Dense(16, activation='tanh', input_shape=(10000,)))
# model.add(layers.Dense(16, activation='tanh')) # different activation function --> minor change in accuracy
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dense(16, activation='relu')) # extra layer --> minor change in accuracy
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model

# model.compile(optimizer='rmsprop',
#               loss='mse',
#               metrics=['accuracy']) # different loss function --> minor change in accuracy
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Set aside a validation set

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# Train the model

# Initial training (20 epochs)

# history = model.fit(partial_x_train,
#                     partial_y_train,
#                     epochs=20,
#                     batch_size=512,
#                     validation_data=(x_val, y_val))

# Ideal training (4 epochs)

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=4,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# Get historical data from model training

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

# Plot training/ validation loss

plt.plot(epochs, loss_values, 'bo', label='Training Loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training/ validation loss

plt.figure()

accuracy = history_dict['acc']
val_accuracy = history_dict['val_acc']

plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Evaluate the test data and get the results

results = model.evaluate(x_test, y_test)
