import matplotlib.pyplot as plt
from Data_Retrieval import CryptoDataRetriever
from calculate_metrics import MetricsCalculator
from ml_model import MLModel
from logger import logging as lg

# Configuration details
API_KEY = 'your api key paste here' 
CRYPTO_PAIR = "BTC/USD"
TIMESTAMP="1MIN"
START_DATE = "2019-01-01"
END_DATE = "2023-12-31"
VARIABLE1 = 30    # look-back period for historical high and low metrics.
VARIABLE2 = 8

def main():
    # Fetch crypto data
    lg.info("main function called successfully")

    retriever = CryptoDataRetriever(API_KEY)
    crypto_data_df = retriever.fetch_crypto_data(CRYPTO_PAIR, START_DATE, END_DATE, TIMESTAMP)
    lg.info("Data Received successfully")

    # Calculate metrics
    calculator = MetricsCalculator()

    result_df = calculator.calculate_metrics(crypto_data_df, VARIABLE1, VARIABLE2)
    def fill_na_with_appropriate_value(df):
        # Count NaN values in each column
        nan_counts = df.isna().sum()
        print("Count of NaN values in each column:")
        print(nan_counts)
    
        # Replace NaN values
        for column in df.columns:
            if df[column].dtype in ['float64', 'int64']:  # Numeric columns
                # Use mean for numeric columns
                mean_value = df[column].mean()
                df[column].fillna(mean_value, inplace=True)
            elif df[column].dtype == 'object':  # Categorical columns
                # Use mode for categorical columns
                mode_value = df[column].mode()[0]
                df[column].fillna(mode_value, inplace=True)
    
    # Call the function to fill NaN values
    clean=fill_na_with_appropriate_value(result_df)
    
    # Display the modified DataFrame
    print("\nModified DataFrame:")
    
    nan_count = result_df.isna().sum()
    print(nan_count)
    
    # Train ML model
    model = MLModel()
    model.train_and_evaluate(result_df, VARIABLE1, VARIABLE2)

if __name__ == "__main__":
    main()
