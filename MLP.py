# Starting the time
import datetime
start = datetime.datetime.now()

# Import required libraries
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential

# Load data and split it into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Plot the data
plt.imshow(X_train[1])
plt.show()
print(y_train[1])

# Process of flattening images
pixels_num = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], pixels_num)).astype('float32') #float32 to save space
X_test = X_test.reshape((X_test.shape[0], pixels_num)).astype('float32') #float32 to save space
print("Number of pixels: ", pixels_num)

# Normalizing data
X_train = X_train / 255
X_test = X_test / 255

# Applying one-hot encoding
y_train = np_utils.to_categorical(y_train) #values will be 0 and 1
y_test = np_utils.to_categorical(y_test) #values will be 0 and 1
classes_num = y_test.shape[1] 
print("number of classes: ", classes_num)

# Baseline model
def baseline_model():
  model_mlp = Sequential()
  model_mlp.add(Dense(pixels_num, input_dim=pixels_num, kernel_initializer='normal', activation='relu')) #kernal is normal
  model_mlp.add(Dense(500, kernel_initializer='normal', activation='relu')) #kernal is normal
  model_mlp.add(Dense(classes_num, kernel_initializer='normal', activation='tanh')) #kernal is normal
  model_mlp.add(Dense(classes_num, kernel_initializer='normal', activation='softmax')) #kernal is normal
  
  model_mlp.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model_mlp

# Build and fit the model
model_mlp = baseline_model()
hist = model_mlp.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=128, verbose=2)

# Performance evaluation
scores = model_mlp.evaluate(X_test, y_test, verbose=0)
print("Accuracy rate: " + str(scores[1]*100))

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
model_mlp.summary()

# Ending the time and printing the execution time
end = datetime.datetime.now() - start
print("Total time required : ", end)
