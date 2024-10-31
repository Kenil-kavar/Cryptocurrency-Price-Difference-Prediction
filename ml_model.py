# ml_model.py
import keras_tuner as kt
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Bidirectional
import pandas as pd
from tensorflow.keras.models import load_model
from plot_keras_history import plot_history
import numpy as np
import os
from plot_keras_history import plot_history
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from logger import logging as lg
from keras.models import Sequential
import matplotlib.pyplot as plt



class MLModel:


    def reshape_data(X, time_steps):
        X_reshaped = []
        for i in range(len(X) - time_steps):
            X_reshaped.append(X[i:(i + time_steps)])
        return np.array(X_reshaped)
        
    def train_and_evaluate(self, result_df, variable1, variable2):
        lg.info("train_and_evaluate method called successfully.")
        result_df.to_excel('DataUsedForTraining.xlsx', index=False)
        result_df.to_csv('DataUsedForTraining.csv', index=False)
        X = result_df[[
            f'Days_Since_High_Last_{variable1}_Days',
            f'%_Diff_From_High_Last_{variable1}_Days',
            f'Days_Since_Low_Last_{variable1}_Days',
            f'%_Diff_From_Low_Last_{variable1}_Days'
        ]].values

        y_high = result_df[f'%_Diff_From_High_Next_{variable2}_Days'].values
        y_low = result_df[f'%_Diff_From_Low_Next_{variable2}_Days'].values

        X_train_high, X_test_high, y_train_high, y_test_high = train_test_split(X, y_high, test_size=0.3, random_state=42)
        X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(X, y_low, test_size=0.3, random_state=42)

        time_steps = 15
        X_train_high = MLModel.reshape_data(X_train_high, time_steps)
        X_test_high = MLModel.reshape_data(X_test_high, time_steps)
        X_train_low = MLModel.reshape_data(X_train_low, time_steps)
        X_test_low = MLModel.reshape_data(X_test_low, time_steps)

        y_train_high = y_train_high[time_steps:]
        y_test_high = y_test_high[time_steps:]
        y_train_low = y_train_low[time_steps:]
        y_test_low = y_test_low[time_steps:]
        def build_model(hp):
            model = Sequential()
            model.add(Input(shape=(X_train_high.shape[1], X_train_high.shape[2])))
        
            # Tune the number of units in each LSTM layer
            for i in range(3):  # Number of LSTM layers
                model.add(Bidirectional(LSTM(
                    units=hp.Int(f'lstm_units_{i}', min_value=32, max_value=256, step=64),
                    return_sequences=(i < 2),  # Only last LSTM layer does not return sequences
                    kernel_regularizer=l1_l2(
                        l1=hp.Float(f'l1_reg_{i}', 1e-6, 1e-3, sampling="log"),
                        l2=hp.Float(f'l2_reg_{i}', 1e-6, 1e-3, sampling="log")
                    )
                )))
                model.add(Dropout(rate=hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.3, step=0.05)))
        
            # Fully connected layer
            model.add(Dense(32, activation='relu'))
            model.add(BatchNormalization())
        
            # Output layer
            model.add(Dense(1))
        
            # Compile the model
            model.compile(
                optimizer=Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-3, sampling='log')),
                loss='mean_squared_error',
                metrics=['mae']
            )
            model.summary()
            
            return model

        tuner_high = kt.RandomSearch(
            build_model,
            objective='val_mae',
            max_trials=20,
            executions_per_trial=1,
            directory='Hypertunning',
            project_name='high_prediction'
        )

        tuner_low = kt.RandomSearch(
            build_model,
            objective='val_mae',
            max_trials=20,
            executions_per_trial=1,
            directory='Hypertunning',
            project_name='low_prediction'
        )
        early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

        tuner_high.search(X_train_high, y_train_high, epochs=10, validation_data=(X_test_high, y_test_high))
        tuner_low.search(X_train_low, y_train_low, epochs=10, validation_data=(X_test_low, y_test_low))

        best_model_high = tuner_high.get_best_models(num_models=1)[0]
        best_model_low = tuner_low.get_best_models(num_models=1)[0]


        history_high = best_model_high.fit(X_train_high, y_train_high,
                                            validation_data=(X_test_high, y_test_high),
                                            epochs=80,
                                            batch_size=32,
                                            callbacks=[early_stopping])
        lg.info("best_model_high trained successfully")
        output_folder = 'Graph'
        os.makedirs(output_folder, exist_ok=True)
        plot_history(history_high)

        # Save the plot to a file
        plt.savefig('Graph/history_high_plot.png')  # Specify the path where you want to save the plot
        
        # Optionally, close the plot if you don't want to display it
        plt.close()



        
        
        # Train the best model and capture the history
        history_low = best_model_low.fit(
            X_train_low, y_train_low,
            validation_data=(X_test_low, y_test_low),
            epochs=80,
            batch_size=32
        )
        lg.info("best_model_low trained successfully")
        
        
        os.makedirs(output_folder, exist_ok=True)
        
        plot_history(history_low)

        # Save the plot to a file
        plt.savefig('Graph/history_low_plot.png')  # Specify the path where you want to save the plot
        
        # Optionally, close the plot if you don't want to display it
        plt.close()
        



        best_model_high.save('model_high.h5')
        lg.info("model_high saved successfully")
        best_model_low.save('model_low.h5')
        lg.info("model_low saved successfully")

        # Load the model
        
        loaded_model_high = load_model('model_high.h5')
        lg.info("model_high loaded successfully")

        loaded_model_low = load_model('model_low.h5')
        lg.info("model_low loaded successfully")


        # Evaluate high prediction model
        y_pred_high = loaded_model_high.predict(X_test_high).flatten()
        y_pred_low = loaded_model_low.predict(X_test_low).flatten()
        

        # Make predictions
        predicted_high = best_model_high.predict(X_test_high).flatten()
        predicted_low = best_model_low.predict(X_test_low).flatten()
        

        
        # Plot predicted vs. actual for high target
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.plot(predicted_high, label="Predicted % Diff From High", color="blue")
        plt.plot(y_test_high, label="Actual % Diff From High", color="orange")
        plt.title("Predicted vs Actual - % Diff From High Next 1 Day")
        plt.xlabel("Test Sample")
        plt.ylabel("% Difference")
        plt.legend()
        plt.savefig(os.path.join(output_folder, 'Pred vs Actual High.png'))
        
        # Plot predicted vs. actual for low target
        plt.subplot(1, 2, 2)
        plt.plot(predicted_low, label="Predicted % Diff From Low", color="black")
        plt.plot(y_test_low, label="Actual % Diff From Low", color="red")
        plt.title("Predicted vs Actual - % Diff From Low Next 1 Day")
        plt.xlabel("Test Sample")
        plt.ylabel("% Difference")
        plt.legend()
        plt.savefig(os.path.join(output_folder, 'Pred vs Actual low.png'))

        
        plt.tight_layout()
        plt.close()



        
        # Calculate MAE, MSE, and R-squared for the high model
        mae_high = mean_absolute_error(y_test_high, y_pred_high)
        mse_high = mean_squared_error(y_test_high, y_pred_high)
        r2_high = r2_score(y_test_high, y_pred_high)
        
        # Calculate MAE, MSE, and R-squared for the low model
        mae_low = mean_absolute_error(y_test_low, y_pred_low)
        mse_low = mean_squared_error(y_test_low, y_pred_low)
        r2_low = r2_score(y_test_low, y_pred_low)
        
        print(f"High Model - MAE: {mae_high}, MSE: {mse_high}, R-squared: {r2_high}")
        print(f"Low Model - MAE: {mae_low}, MSE: {mse_low}, R-squared: {r2_low}")
        
        # Optional: Calculate Percentage Accuracy for a specified tolerance margin (e.g., 10%)
        tolerance = 0.10# 10% tolerance
        
        # Define a function to calculate percentage accuracy within tolerance
        def percentage_accuracy(y_true, y_pred, tolerance):
            within_tolerance = np.abs((y_true - y_pred) / y_true) < tolerance
            return np.mean(within_tolerance) * 100  # Percentage of predictions within tolerance
        
        # Calculate percentage accuracy for both models
        accuracy_high = percentage_accuracy(y_test_high, y_pred_high, tolerance)
        accuracy_low = percentage_accuracy(y_test_low, y_pred_low, tolerance)
        
        print(f"High Model - Percentage Accuracy within {tolerance * 100}%: {accuracy_high}%")
        print(f"Low Model - Percentage Accuracy within {tolerance * 100}%: {accuracy_low}%")
        
        
        
