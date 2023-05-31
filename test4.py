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
        self.model = self.build_model()
    
    def build_model(self):
        model = Sequential()
        model.add(LSTM(units=64, input_shape=self.input_shape, return_sequences=True))
        model.add(LSTM(units=32, return_sequences=False))
        model.add(Dense(units=self.output_shape, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    
    def train(self, X_train, y_train, X_test, y_test, num_epochs, batch_size):
        # update with your_path
        save_fname = os.path.join('your_path', '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(num_epochs)))

        callbacks = [EarlyStopping(monitor='val_loss', patience=50),
                     ModelCheckpoint(filepath=save_fname, monitor='val_loss',save_best_only=True)]
        history = self.model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1,
                             validation_data=(X_test, y_test), callbacks=callbacks)
        self.model.save(save_fname)
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

model = LSTM_model((look_back, num_features), 1)

epochs = 10
batch_size = 64

history = model.train(X_train, y_train, X_test, y_test, epochs, batch_size)

predictions = model.predict(X_test)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print("Mean, std deviation")
print("%.2f%% (+/- %.2f%%)" % (np.mean(loss)*100, np.std(loss)*100))

# Plot history
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# calculate RMSE
rmse = sqrt(mean_squared_error(y_test, predictions))
print('Test RMSE: %.3f' % rmse)

difference= abs(y_test - predictions)
median_value = np.median(difference)
mean_value = np.mean(difference)

# Plot difference, mean, median
plt.plot(difference, label='Difference')
plt.axhline(mean_value, color='r', linestyle='--', label='Difference Mean Value')
plt.axhline(median_value, color='g', linestyle='--', label='Difference Median')
plt.xlabel('Stop_order')
plt.ylabel('Difference')
plt.legend()

plt.text(0, mean_value, f'Mean: {mean_value: .3f}', color='r', ha='right', va='bottom')
plt.text(0, median_value, f'Median: {median_value:.3f}', color='g', ha='right', va='top')
plt.show() 

# Plot actual vs predicted values
plt.plot(y_test, label='Test Data')
plt.plot(predictions, label='Predictions')

plt.xlabel('Stop_order')
plt.ylabel('Ridership')
plt.legend()

plt.show()
