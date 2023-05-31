import os
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

m = 3 # previous stops
n = 2 # previous days
num_line_descr = "1"

dataset_folder_path = "LSTM_Dataset_" + num_line_descr

input_sequence = pd.read_csv(os.path.join(dataset_folder_path, 'inputs.csv'), delimiter=',') 
target_sequence = pd.read_csv(os.path.join(dataset_folder_path, 'targets.csv'))

num_features = input_sequence.shape[1]
num_samples = target_sequence.shape[0]
look_back = m + n

assert (num_samples * 5) == input_sequence.shape[0], "Wrong LSTM Dataset"

X = np.zeros((num_samples, look_back, num_features))
y = np.zeros((num_samples, 1))

for i in range(0, input_sequence.shape[0], look_back):
    idx = int(i/look_back)
    X[idx,:,:] = input_sequence.iloc[i:i+look_back,:]
    y[idx, 0] = target_sequence.iloc[idx].item()
        
# Reshape X to 2D (num_samples, look_back * num_features)
#X_2d = X.reshape(X.shape[0], -1)

# Reshape y to 1D (num_samples,)
#y_1d = y.reshape(-1)

# Perform train-test split
#X_train, X_test, y_train, y_test = train_test_split(X_2d, y_1d, test_size=0.2, random_state=42)

# Define the train-test split ratio
test_size = 0.2 

# Calculate the number of samples for testing
num_test_samples = int(test_size * num_samples)

# Shuffle the data indices
indices = np.arange(num_samples)
np.random.shuffle(indices)

# Split the indices into train and test sets
train_indices = indices[:-num_test_samples]
test_indices = indices[-num_test_samples:]

# Split the data based on the indices
X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]
