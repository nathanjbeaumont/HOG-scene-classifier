import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
import numpy as np

database_path = Path(__file__).parents[1] / 'archive'
with open(database_path / 'labels.pkl', "rb") as fp:
    label_list = pickle.load(fp)

hog_matrix = np.load(database_path / 'hog_matrix.npy')

print(hog_matrix.shape)

random_indeces = np.random.permutation(np.arange(hog_matrix.shape[0]))
random_indeces_train = random_indeces[:500]
random_indeces_test = random_indeces[500:1000]

hog_matrix_train = hog_matrix[random_indeces_train, :]
hog_matrix_test = hog_matrix[random_indeces_test, :]

labels_train = [label_list[i] for i in random_indeces_train]
labels_test = [label_list[i] for i in random_indeces_test]

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(hog_matrix_train, labels_train)
print(f"Accuracy: {clf.score(hog_matrix_test, labels_test)}")
