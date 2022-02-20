# Starting the time
import datetime
start = datetime.datetime.now()

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import scale
from random import shuffle
import gc

# Loading MNIST data 
train = pd.read_csv('mnist_train.csv')
test = pd.read_csv('mnist_test.csv')

# Dimensions
print("Dimensions: ",train.shape)
print("\nDimensions: ",test.shape, "\n")

# Visualizing the dataset
plt.plot(figure = (16,10))
g = sns.countplot( train["label"], palette = 'icefire')
plt.title('Digit classes')
train.label.astype('category').value_counts()

# Plotting samples
sample = train.iloc[3, 1:]
sample = sample.values.reshape(28,28)
plt.imshow(sample, cmap='gray')
plt.title("Digit 1")

# Shuffeling 
shuffle(train.values)

# Feature selection 
X_train = train.drop(labels = ["label"],axis = 1) 
y_train = train["label"]

# Test data 
X_test = test.values

print(f'X_train = {X_train.shape}, y = {y_train.shape}, X_test = {X_test.shape}')

# Plotting digits
plt.figure(figsize=(14,12))
for digit_number in range(0,30):
    plt.subplot(7,10,digit_number+1)
    data_reshape = X_train.iloc[digit_number].values.reshape(28,28)  # reshaping to 2d pixel array
    plt.imshow(data_reshape, interpolation = "none", cmap = "afmhot")
    plt.xticks([])
    plt.yticks([])
plt.tight_layout() #to keep everything fitting well

# class distribution 
sns.set(style="darkgrid")
counts = sns.countplot(x="label", data=train, palette="Set1")

# average feature values
round(train.drop('label', axis=1).mean(), 2)

# Deciding X and y
X = train.drop(columns = 'label')
y = train['label']

# Printing the size of data 
print(train.shape)

# Normalizing data
X = X / 255
test = test / 255

print("X:", X.shape)
print("Test_data: ", test.shape)

# Scaling
X_scaled = scale(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, train_size = 0.2 ,random_state = 10)

# SVM model
model_svm = SVC(kernel='linear') #kernal is linear
model_svm.fit(X_train, y_train)

# Prediction
X_pred = model_svm.predict(X_train)
y_pred = model_svm.predict(X_test)

# Performance evaluation
predictions = model_svm.predict(X_train)
metrics.accuracy_score(predictions, y_train)

print(metrics.accuracy_score(y_train, X_pred, normalize=True, sample_weight=None))
print(metrics.confusion_matrix(y_true=y_train, y_pred=X_pred))
print("Accuracy: ", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")
print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))

# Summary of the model
model_svm.summary()

# Ending the time and printing the execution time
end = datetime.datetime.now() - start
print("Total time required : ", end)

# The point from this code is to free some memory as the SVM model is too heavy
# and without running this code we will have memory error.
gc.collect()
