from pathlib import Path
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

database_path = Path(__file__).parents[1] / 'archive'
with open(database_path / 'labels.pkl', "rb") as fp:
    label_list = pickle.load(fp)

hog_matrix = np.load(database_path / 'hog_matrix.npy')

# Encode the labels
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(label_list)
categorical_labels = to_categorical(encoded_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(hog_matrix, categorical_labels, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(8100,), kernel_regularizer=l2(0.001)))
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dense(10, activation='softmax'))  # Number of unique classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Configure early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=5, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate the model
_, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")