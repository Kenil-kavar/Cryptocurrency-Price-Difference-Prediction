import pandas as pd
from logger import logging as lg

class MetricsCalculator:

    def calculate_metrics(self, data: pd.DataFrame, variable1: int, variable2: int) -> pd.DataFrame:
        lg.info("calculate_metrics method called successfully")
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')

        # Calculating Historical High Price (Column: High_Last_{variable1}_Days)
        data[f'High_Last_{variable1}_Days'] = data['High'].rolling(window=variable1, min_periods=1).max()
    
        # Calculating Historical Low Price (Column: Low_Last_{variable1}_Days)
        data[f'Low_Last_{variable1}_Days'] = data['Low'].rolling(window=variable1, min_periods=1).min()
    
        # Initialize Series for Last High and Low Dates
        last_high_dates = pd.Series(index=data.index, dtype='datetime64[ns]')
        last_low_dates = pd.Series(index=data.index, dtype='datetime64[ns]')
    
        for i in range(len(data)):
            # If the high price on day i is the maximum within the 'variable1' day window, store that date
            last_high_dates.iloc[i] = data.index[i] if data['High'].iloc[max(0, i-variable1+1):i+1].max() == data['High'].iloc[i] else last_high_dates.iloc[i-1]
            # If the low price on day i is the minimum within the 'variable1' day window, store that date
            last_low_dates.iloc[i] = data.index[i] if data['Low'].iloc[max(0, i-variable1+1):i+1].min() == data['Low'].iloc[i] else last_low_dates.iloc[i-1]

        # Calculate the number of days since the last high and low prices for each row
        data[f'Days_Since_High_Last_{variable1}_Days'] = (data.index - last_high_dates).dt.days
        data[f'Days_Since_Low_Last_{variable1}_Days'] = (data.index - last_low_dates).dt.days

        # Calculate the percentage difference between the close price and the historical high over 'variable1' days
        data[f'%_Diff_From_High_Last_{variable1}_Days'] = ((data['Close'] - data[f'High_Last_{variable1}_Days']) / data[f'High_Last_{variable1}_Days']) * 100
        # Calculate the percentage difference between the close price and the historical low over 'variable1' days
        data[f'%_Diff_From_Low_Last_{variable1}_Days'] = ((data['Close'] - data[f'Low_Last_{variable1}_Days']) / data[f'Low_Last_{variable1}_Days']) * 100

        # Calculate the high price for the next 'variable2' days
        data[f'High_Next_{variable2}_Days'] = data['High'].shift(-variable2).rolling(window=variable2, min_periods=1).max()
        # Calculate the percentage difference between the close price and the high over the next 'variable2' days
        data[f'%_Diff_From_High_Next_{variable2}_Days'] = ((data['Close'] - data[f'High_Next_{variable2}_Days']) / data[f'High_Next_{variable2}_Days']) * 100                                                                       
        data[f'Low_Next_{variable2}_Days'] = data['Low'].shift(-variable2).rolling(window=variable2, min_periods=1).min()                                                                                                                                                                                                                                                       
        data[f'%_Diff_From_Low_Next_{variable2}_Days'] = ((data['Close'] - data[f'Low_Next_{variable2}_Days']) / data[f'Low_Next_{variable2}_Days']) * 100

        data = data.reset_index()    # Reseting the index of the DataFrame to make 'Date' a regular column again
        lg.info("Data calculated successfully.")
        return data
