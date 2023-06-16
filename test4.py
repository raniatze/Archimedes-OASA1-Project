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

    def train(self, X_train, y_train, epochs, batch_size, validation_data, category):
        
        # Define the directory path for saving the checkpoints
        checkpoint_dir = os.path.join(os.getcwd(), 'Checkpoints/Category_{category}_{epochs}_{batch_size}'.format(category=category, epochs=epochs, batch_size=batch_size))

        # Create the directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
    
        checkpoint_callback = ModelCheckpoint(os.path.join(checkpoint_dir, 'best_model.h5'), monitor='val_loss', mode='min', save_best_only=True)
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
look_back = m + n

category_0 = ['0', '1', '139', '140', '4', '5', '143', '6', '144', '8', '9', '188', '145', '191', '172', '15', '20', '173', '22', '205']

category_1 = ['2', '217', '10', '12', '16', '18', '174', '193', '33', '39', '194', '58', '68', '70', '225', '90', '92', '179', '113']

category_2 = ['11', '189', '13', '190', '192', '14', '181', '146', '307', '19', '21', '200', '209', '24', '175', '148', '182', '210', '38', '40']

category_3 = ['114', '115', '116', '169', '117', '118', '170', '119', '120', '230', '122', '123', '124', '125', '199', '127', '128', '129', '130']

category_4 = ['171', '17', '49', '50', '152', '65', '74', '82', '196', '107', '108']

category_5 = ['142', '37', '226', '223', '331', '332', '237', '206', '204', '231', '131', '132', '133', '134', '135', '244']

X, y = [], []

category = '4' # CHANGE accordingly
for num_line_descr in category_4: # CHANGE accordingly

  dataset_folder_path = "Category_" + category +  "/LSTM_Dataset_" + num_line_descr
  print(dataset_folder_path)

  input_sequence = pd.read_csv(os.path.join(dataset_folder_path, 'inputs.csv'), delimiter=',')
  target_input_sequence = pd.read_csv(os.path.join(dataset_folder_path, 'targets.csv'), delimiter=',')

  target_sequence = target_input_sequence['T_pa_in_veh']

  num_features = input_sequence.shape[1]
  num_samples = target_sequence.shape[0]

  assert (num_samples * 5) == input_sequence.shape[0], "Wrong LSTM Dataset"

  X_line_descr = np.zeros((num_samples, look_back+1, num_features))
  y_line_descr = np.zeros((num_samples, 1))

  for i in range(0, input_sequence.shape[0], look_back):
    
    idx = int(i/look_back)
    X_line_descr[idx,:-1,:] = input_sequence.iloc[i:i+look_back,:]
    pred_row = np.array(target_input_sequence.iloc[idx,:])
    pred_row[12] = -1
    X_line_descr[idx,-1,:] = pred_row
    y_line_descr[idx, 0] = target_sequence.iloc[idx].item()
  
  X.append(X_line_descr)
  y.append(y_line_descr)

X = np.concatenate(X, axis=0)
y = np.vstack(y)

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
np.random.seed(0)
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

epochs = 100
batch_size = 128
history = model.train(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), category=category)

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

# Define the directory path for saving the plots
plots_dir = os.path.join(os.getcwd(), 'Plots/Category_{category}'.format(category=category))

# Create the directory if it doesn't exist
os.makedirs(plots_dir, exist_ok=True)

file = os.path.join(plots_dir, str(epochs) + '_' + str(batch_size) + "_0.jpg")
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

file = os.path.join(plots_dir, str(epochs) + '_' + str(batch_size) + "_1.jpg")
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


file = os.path.join(plots_dir, str(epochs) + '_' + str(batch_size) + "_2.jpg")
with open(file,'w') as f:
    pass
plt.savefig(file, format='jpg')
