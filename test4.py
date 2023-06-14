import os
import csv
import numpy as np
import pandas as pd
import datetime as dt
from math import sqrt
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

class LSTM_model:

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units=64, input_shape=self.input_shape, return_sequences=True))
        self.model.add(LSTM(units=32, return_sequences=False))
        self.model.add(Dense(units=self.output_shape, activation='linear'))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def train(self, X_train, y_train, epochs, batch_size, validation_data):
        
        # Define the directory path for saving the checkpoints
        checkpoint_dir = os.path.join(os.getcwd(), 'Checkpoints', num_line_descr)

        # Create the directory if it doesn't exist
        os.makedirs(checkpoint_dir)
    
        checkpoint_callback = ModelCheckpoint(os.path.join(checkpoint_dir, 'model_checkpoint_{epoch:02d}.h5'), save_freq='epoch', save_best_only=False)
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1,
                             validation_data=validation_data, callbacks=[checkpoint_callback, early_stopping_callback])
        return history

    def evaluate(self, X_test, y_test):
        loss = self.model.evaluate(X_test, y_test)
        return loss

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

m = 3 # previous stops
n = 2 # previous days
num_line_descr = "1" # specific line description

dataset_folder_path = "LSTM_Dataset_" + num_line_descr

input_sequence = pd.read_csv(os.path.join(dataset_folder_path, 'inputs.csv'), delimiter=',')
target_input_sequence = pd.read_csv(os.path.join(dataset_folder_path, 'targets.csv'), delimiter=',')
print(input_sequence.shape)
print(target_input_sequence.shape)
x = input()

target_sequence = target_input_sequence['T_pa_in_veh']

num_features = input_sequence.shape[1]
num_samples = target_sequence.shape[0]
look_back = m + n

assert (num_samples * 5) == input_sequence.shape[0], "Wrong LSTM Dataset"

X = np.zeros((num_samples, look_back+1, num_features))
y = np.zeros((num_samples, 1))

for i in range(0, input_sequence.shape[0], look_back):
    idx = int(i/look_back)
    X[idx,:-1,:] = input_sequence.iloc[i:i+look_back,:]
    pred_row = np.array(target_input_sequence.iloc[idx,:])
    pred_row[12] = -1
    X[idx,-1,:] = pred_row
    y[idx, 0] = target_sequence.iloc[idx].item()

    print(X)
    print(y)

# Reshape X to 2D (num_samples, look_back * num_features)
#X_2d = X.reshape(X.shape[0], -1)

# Reshape y to 1D (num_samples,)
#y_1d = y.reshape(-1)

# Perform train-test split
#X_train, X_test, y_train, y_test = train_test_split(X_2d, y_1d, test_size=0.2, random_state=42)

# Define the train-validation-test split ratios
validation_size = 0.1
test_size = 0.2

# Calculate the number of samples for validation and testing
num_validation_samples = int(validation_size * num_samples)
num_test_samples = int(test_size * num_samples)

# Shuffle the data indices
indices = np.arange(num_samples)
np.random.shuffle(indices)

# Split the indices into train, validation, and test sets
train_indices = indices[:-(num_validation_samples + num_test_samples)]
val_indices = indices[-(num_validation_samples + num_test_samples):-num_test_samples]
test_indices = indices[-num_test_samples:]

# Split the data based on the indices
X_train = X[train_indices]
y_train = y[train_indices]
X_val = X[val_indices]
y_val = y[val_indices]
X_test = X[test_indices]
y_test = y[test_indices]

model = LSTM_model((look_back+1, num_features), 1)

history = model.train(X_train, y_train, epochs = 20, batch_size = 128, validation_data=(X_val, y_val))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print("Mean, std deviation")
print("%.2f%% (+/- %.2f%%)" % (np.mean(loss)*100, np.std(loss)*100))

# Make predictions
predictions = model.predict(X_test)

# Plot history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

directory = './'
file = directory + str(epochs) + '_' + str(batch_size) + "_0.jpg"
with open(file,'w') as f:
    pass
plt.savefig(file, format='jpg')

# calculate RMSE
rmse = sqrt(mean_squared_error(y_test, predictions))
print('Test RMSE: %.3f' % rmse)

difference= abs(y_test - predictions)
median_value = np.median(difference)
mean_value = np.mean(difference)

# Plot difference, mean, median
plt.clf()
plt.plot(difference, label='Difference')
plt.axhline(mean_value, color='r', linestyle='--', label='Difference Mean Value')
plt.axhline(median_value, color='g', linestyle='--', label='Difference Median')
plt.xlabel('Stop_order')
plt.ylabel('Difference')
plt.legend()

plt.text(0, mean_value, f'Mean: {mean_value: .3f}', color='r', ha='right', va='bottom')
plt.text(0, median_value, f'Median: {median_value:.3f}', color='g', ha='right', va='top')

file = directory + str(epochs) + '_' + str(batch_size) + "_1.jpg"
with open(file,'w') as f:
    pass
plt.savefig(file, format='jpg')

# Plot actual vs predicted values
plt.clf()
plt.plot(y_test, label='Test Data')
plt.plot(predictions, label='Predictions')
plt.xlabel('Stop_order')
plt.ylabel('Ridership')
plt.legend()


file = directory + str(epochs) + '_' + str(batch_size) + "_2.jpg"
with open(file,'w') as f:
    pass
plt.savefig(file, format='jpg')
