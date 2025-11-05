import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ml_training.ml_lib.neighbors import KNeighborsClassifier
from ml_training.ml_lib.model_selection import train_test_split
from ml_training.ml_lib.metrics import accuracy_score

df_churn = pd.read_csv('D:/downloads/telecom_churn_clean.csv')

X = df_churn[['account_length', 'customer_service_calls']]
y = df_churn['churn']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X_train, y_train)
predictions = knn1.predict(X_test)

knn1accuracy = accuracy_score(y_test, predictions)
print(f"Knn1 accuracy: {knn1accuracy}")

knn2 = KNeighborsClassifier(n_neighbors=2)
knn2.fit(X_train, y_train)
predictions = knn2.predict(X_test)

knn2accuracy = accuracy_score(y_test, predictions)
print(f"Knn2 accuracy: {knn2accuracy}")

knn4 = KNeighborsClassifier(n_neighbors=4)
knn4.fit(X_train, y_train)
predictions = knn4.predict(X_test)

knn4accuracy = accuracy_score(y_test, predictions)
print(f"Knn4 accuracy: {knn4accuracy}")

knn8 = KNeighborsClassifier(n_neighbors=8)
knn8.fit(X_train, y_train)
predictions = knn8.predict(X_test)

knn8accuracy = accuracy_score(y_test, predictions)
print(f"Knn8 accuracy: {knn8accuracy}")

knn16 = KNeighborsClassifier(n_neighbors=16)
knn16.fit(X_train, y_train)
predictions = knn16.predict(X_test)

knn16accuracy = accuracy_score(y_test, predictions)
print(f"Knn16 accuracy: {knn16accuracy}")

neighbors = [1, 2, 4, 8, 16]

# Test accuracy values you already calculated
test_acc = [
    knn1accuracy, knn2accuracy, knn4accuracy, knn8accuracy, knn16accuracy
]

# Plot
plt.figure(figsize=(8, 5))
plt.plot(neighbors, test_acc, marker='o', color='blue', label='Test Accuracy')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('KNN: Number of Neighbors vs Test Accuracy (Unscaled)')
plt.xticks(neighbors)
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.show()
