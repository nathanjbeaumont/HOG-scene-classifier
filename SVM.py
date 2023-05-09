import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

hog_matrix = np.load("hog_matrix.npy")

with open("labels.pkl", "rb") as f:
    label_list = pickle.load(f)

parameters = {'kernel':('linear', 'rbf'), 'C':[0.5, 2, 5, 10]}


#Encode the labels
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(label_list)

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(hog_matrix, encoded_labels, test_size=0.2, random_state=42)

#SVM (defaults to gaussian)
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(hog_matrix, label_list)

#Make predictions on the test set
y_pred = clf.predict(X_test)

#Evaluate the classifier
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {test_accuracy:.4f}")
