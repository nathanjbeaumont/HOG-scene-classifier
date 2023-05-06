from pathlib import Path
import pickle

import numpy as np

database_path = Path(__file__).parents[1] / 'archive'
with open(database_path / 'labels.pkl', "rb") as fp:
    label_list = pickle.load(fp)

hog_matrix = np.load(database_path / 'hog_matrix.npy')

print(hog_matrix.shape)
print(len(label_list))
