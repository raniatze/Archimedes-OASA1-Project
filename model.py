from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from keras.wrappers.scikit_learn import KerasRegressor

class LSTM_model:

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = self.build_model()

    def build_model(self, units_lstm=64, units_dense=32, activation='linear', optimizer='adam'):
        model = Sequential()
        model.add(LSTM(units=units_lstm, input_shape=self.input_shape, return_sequences=True))
        model.add(LSTM(units=units_lstm, return_sequences=False))
        model.add(Dense(units=units_dense, activation=activation))
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    def train(self, X, y, num_epochs, batch_size):
        regressor = KerasRegressor(build_fn=self.build_model, verbose=0)
        param_grid = {
            'units_lstm': [32, 64, 128],                        # Example LSTM units values to search
            'units_dense': [16, 32, 64],                         # Example Dense units values to search
            'activation': ['linear', 'relu', 'sigmoid'],         # Example activation functions to search
            'optimizer': ['adam', 'rmsprop', 'sgd']              # Example optimizers to search
        }
        grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=TimeSeriesSplit(n_splits=3))
        grid_search.fit(X, y)
        best_params = grid_search.best_params_

        self.model = self.build_model(units_lstm=best_params['units_lstm'], units_dense=best_params['units_dense'],
                                      activation=best_params['activation'], optimizer=best_params['optimizer'])
        
        save_fname = os.path.join('your_path', '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(num_epochs)))

        callbacks = [EarlyStopping(monitor='val_loss', patience=50),
                     ModelCheckpoint(filepath=save_fname, monitor='val_loss',save_best_only=True)]
        history = self.model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1,
                             validation_data=(X_test, y_test), callbacks=callbacks)
        self.model.save(save_fname)
        return history


        return history
        
    def evaluate(self, X_test, y_test):
        loss = self.model.evaluate(X_test, y_test)
        return loss

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

