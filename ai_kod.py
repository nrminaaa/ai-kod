Bilet 1;

from bisect import bisect_left
from gzip import BadGzipFile
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Data yarad?lmas? v? modelin qurulmas?
X, y = make_regression(n_samples=100, n_features=1, noise=15.0, random_state=42)
model = LinearRegression().fit(X, y)

# Qiym?tl?ndirm? (MSE v? RMSE)
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

print(f"MSE: {mse:.4f}\nRMSE: {rmse:.4f}")


Bilet 2;

import pandas as pd

# Create the DataFrame
data = {
    'age': [20, 21, 22],
    'score': [70, 80, 90]
}
df = pd.DataFrame(data)

# Show the first 2 rows
print(df.head(2))



Bilet 3;


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2. Perform train_test_split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_split=0.2, random_state=42)

# Verify the split shapes
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


Bilet 4;

import pandas as pd

# Create a pandas Series from the list
s = pd.Series([1, None, 3])

# Fill the None (NaN) value with 0
s_filled = s.fillna(0)

print(s_filled)



Bilet 5;

import numpy as np
from sklearn.datasets import load_iris

# 1. Load the dataset
iris = load_iris()
y = iris.target

# 2. Count how many samples of each class are in y
classes, counts = np.unique(y, return_counts=True)

# 3. Print the results
for cls, count in zip(classes, counts):
    print(f"Class {cls} ({iris.target_names[cls]}): {count} samples")



Bilet 6;


y_true = [1, 0, 1, 1]
y_pred = [1, 1, 1, 0]

# Calculate the number of correct predictions
correct_predictions = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)

# Calculate the total number of predictions
total_predictions = len(y_true)

# Calculate accuracy
accuracy = correct_predictions / total_predictions

print(f"Accuracy: {accuracy}")



Bilet 7;

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Data yükl?nilir v? bölünür
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Model öyr?dilir v? birba?a d?qiqlik (accuracy) hesablan?r
model = LogisticRegression(max_iter=200).fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

print(f"Test Accuracy: {accuracy:.4f}")



Bilet 8;

from sklearn.metrics import confusion_matrix

y_true = [1, 0, 1, 1]
y_pred = [1, 1, 1, 0]

# Qar???ql?q matrisinin hesablanmas?
cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
print(cm)


Bilet 9;

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load dataset and split into train/test sets
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Initialize and train the SVC model with a linear kernel
model = SVC(kernel="linear").fit(X_train, y_train)

# Calculate and print the test accuracy
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")


Bilet 10;

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load and split dataset (80% train, 20% test)
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train Decision Tree and print accuracy directly
clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
print(f"Test Accuracy: {clf.score(X_test, y_test):.4f}")



Bilet 11;

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# 2. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 3. Train the model and print the test accuracy
model = LogisticRegression().fit(X_train, y_train)
print(f"Test Accuracy: {model.score(X_test, y_test):.4f}")


Bilet 12; 

import pandas as pd

# Create the DataFrame
df = pd.DataFrame({'height': [150, 160, 170], 'weight': [50, 60, 70]})

# Show the last 2 rows
print(df.tail(2))



Bilet 13;

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Load the diabetes dataset
X, y = load_diabetes(return_X_y=True)

# Split into train (75%) and test (25%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


Bilet 14;

import pandas as pd

# Create the Series
s = pd.Series([5, None, 7, None])

# Fill missing values with the mean (which is (5 + 7) / 2 = 6.0)
print(s.fillna(s.mean()))


Bilet 15;

from sklearn.datasets import load_wine

# Load the wine dataset
wine = load_wine()

# Get samples and features from the shape of the data matrix
samples, features = wine.data.shape

print(f"Samples: {samples}")
print(f"Features: {features}")


Bilet 16;

y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

# TP: h?m h?qiqi, h?m d? t?xmin 1 olanlar
tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
# Pozitiv t?xminl?rin ümumi say? (TP + FP)
pred_positives = sum(p == 1 for p in y_pred)

precision = tp / pred_positives if pred_positives > 0 else 0
print(f"Precision: {precision}")


Bilet 17;

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

knn = KNeighborsClassifier().fit(X_train, y_train)
print(f"Test Accuracy: {knn.score(X_test, y_test):.4f}")



Bilet 18;

from sklearn.metrics import accuracy_score, f1_score

y_true = [1, 1, 0, 0]
y_pred = [1, 0, 0, 0]

# Calculate and print the metrics
print(f"Accuracy Score: {accuracy_score(y_true, y_pred)}")
print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")



Bilet 19;

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load and split dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train DecisionTree and print test accuracy
clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
print(f"Test Accuracy: {clf.score(X_test, y_test):.4f}")



Bilet 20;

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load and split dataset (70% train, 30% test)
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Train Logistic Regression and print test accuracy
model = LogisticRegression(max_iter=500).fit(X_train, y_train)
print(f"Test Accuracy: {model.score(X_test, y_test):.4f}")