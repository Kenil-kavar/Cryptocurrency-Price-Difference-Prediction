# Cryptocurrency Price Difference Prediction

- This deep learning project aims to predict future price differences in cryptocurrency based on recent historical data, using a bi-directional LSTM model optimized through hyperparameter tuning.

### 📌 Project Overview:-
This project leverages LSTM layers to analyze historical cryptocurrency data, capturing temporal dependencies and using calculated metrics to predict:

% Difference from High in the Next {variable2} Days
% Difference from Low in the Next {variable2} Days
With careful data engineering, model tuning, and a logging mechanism, the project produces reliable predictions to assist in understanding future trends in cryptocurrency pricing.

### 🛠️ Project Structure & Approach

#### Data Retrieval
- Data_Retrieval.py: Retrieves cryptocurrency data via CoinAPI, fetching historical OHLC data efficiently within the user-defined date range.
  
#### Feature Engineering & Metrics Calculation

calculate_metrics.py: Calculates key metrics like historical high and low prices over custom time windows, setting up valuable inputs for the model.

#### Model Building & Training
ml_model.py: Contains the MLmodel class, building and training two LSTM models:
- model_high for predicting % Difference from High
- model_low for predicting % Difference from Low
- Hyperparameters like LSTM layer size, dropout rates, and learning rate are tuned for optimal performance using Keras-Tuner.

#### Logging & Error Handling

logger.py: Provides timestamped logs to track execution and troubleshoot any runtime issues.

### 🚀 Getting Started
Prerequisites
Ensure you have Conda and Python 3.11.10 installed.

Installation
Clone the Repository:
```
git clone https://github.com/Kenil-kavar/Cryptocurrency-Price-Difference-Prediction.git
```
```
cd Cryptocurrency-Price-Difference-Prediction
```
Set Up Environment:
```
conda create -p env python=3.11.10 -y
conda activate ./env
```
Install Dependencies:
```
pip install -r requirements.txt
```
Configuration
- In app.py, set the following values as per your requirements:


API_KEY = "Your CoinAPI Key"

CRYPTO_PAIR = "<your_crypto_pair>"

TIMESTAMP = "<timestamp>"

START_DATE = "<YYYY-MM-DD>"

END_DATE = "<YYYY-MM-DD>"

VARIABLE1 = (for eg 20)

VARIABLE2 = (for eg 7)

### Run the Project
```
python app.py
```
### 📊 Output Files:-
- Logs: Track project execution in the logs folder.
- Graphs: Performance graphs are automatically saved in the Graph folder.

### 🧩 Challenges Faced:-
- API Data Retrieval Limitations: Adjusted data retrieval to accommodate API restrictions by fetching yearly data.
- Model Selection: Experimented with RNN, LSTM, and GRU; Bidirectional LSTM provided optimal results.
- Regularization Tuning: Chose ElasticNet after testing multiple regularizers, achieving the best performance.
- Dual Model Training: Implemented separate models for predicting high and low price differences, enhancing overall accuracy.
