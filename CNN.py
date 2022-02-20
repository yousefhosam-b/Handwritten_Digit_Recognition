# Starting the time
import datetime
start = datetime.datetime.now()

# Import required libraries
import keras
import tensorflow as tf 
import matplotlib.pyplot as plt

# Load data and split it into train and test sets
mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Plot the data
plt.imshow(X_train[1]) 
plt.show()
print(y_train[1])

# Data reshaping
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1) #28 * 28
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1) #28 * 28
input_shape = (28, 28, 1)

# Normalizing data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255
X_test = X_test / 255
print(y_train[0])

# Applying one-hot encoding
y_train = keras.utils.np_utils.to_categorical(y_train, 10) #values will be 0 and 1
y_test = keras.utils.np_utils.to_categorical(y_test, 10) #values will be 0 and 1
print(y_train[0])

# Build and fit the model (takes much time to finish)
model_cnn = keras.models.Sequential()
model_cnn.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=input_shape)) #activation is relu

model_cnn.add(keras.layers.Conv2D(64, (3, 3), activation='relu')) #activation is relu
model_cnn.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model_cnn.add(keras.layers.Dropout(0.25))
model_cnn.add(keras.layers.Flatten())
model_cnn.add(keras.layers.Dense(128, activation='relu')) #activation is relu
model_cnn.add(keras.layers.Dropout(0.5))
model_cnn.add(keras.layers.Dense(10, activation='softmax')) #activation is softmax

model_cnn.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

hist=model_cnn.fit(X_train, y_train,
          batch_size=128,
          epochs=30,
          validation_data=(X_test, y_test))

# Performance evaluation
print(model_cnn.evaluate(X_test, y_test))

training_acc=hist.history['accuracy']
testing_acc=hist.history['val_accuracy']
training_error=hist.history['loss']
testing_error=hist.history['val_loss']
xc=range(30)

# Performance evaluation as a plot (Loss)
plt.figure(1,figsize=(7,5))
plt.plot(xc,training_error)
plt.plot(xc,testing_error)
plt.xlabel('Number of Epochs ---->')
plt.ylabel('Loss ---->')
plt.title('Training_loss vs Testing_loss')
plt.legend(['Training','Testing'])
plt.style.use(['classic'])

# Performance evaluation as a plot (Accuracy)
plt.figure(2,figsize=(7,5))
plt.plot(xc,training_acc)
plt.plot(xc,testing_acc)
plt.xlabel('Num of Epochs ---->')
plt.ylabel('Accuracy ---->')
plt.title('Training_acc vs Testing_acc')
plt.legend(['Training','Validation'],loc=4)
plt.style.use(['classic'])

# Summary of the model
model_cnn.summary()

# Ending the time and printing the execution time
end = datetime.datetime.now() - start
print("Total time required : ", end)

