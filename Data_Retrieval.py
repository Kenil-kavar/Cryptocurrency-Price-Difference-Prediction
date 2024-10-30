
import requests
import pandas as pd
from datetime import datetime, timedelta
from logger import logging as lg

class CryptoDataRetriever:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            'X-CoinAPI-Key': self.api_key
        }

    def fetch_crypto_data(self, crypto_pair, start_date, end_date, timestamp):
        lg.info("fetch_crypto_data method called.")
        def date_ranges(start, end, interval_years=2):
            ranges = []
            current_start = start
            while current_start < end:
                current_end = min(current_start + timedelta(days=interval_years * 365), end)
                ranges.append((current_start, current_end))
                current_start = current_end + timedelta(days=1)
            lg.info("data_ranges function executed successfully")
            return ranges

        pairs_url = "https://rest.coinapi.io/v1/symbols"
        response = requests.get(pairs_url, headers=self.headers)
        lg.info(" Maked API KEY response")

        if response.status_code == 200:
            lg.info("response status code 200 matched")
            pairs_data = response.json()
            symbol_id = None

            # Loop through available pairs to find the matching crypto pair symbol ID
            for pair in pairs_data:
                if pair['asset_id_base'] + "/" + pair['asset_id_quote'] == crypto_pair:
                    symbol_id = pair['symbol_id']
                    break

            if not symbol_id:
                print("Crypto pair not found!")
                return None

            # Initialize an empty DataFrame to hold all fetched data
            all_data = pd.DataFrame()
            start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
            intervals = date_ranges(start_date_dt, end_date_dt)

            # Iterate over each date range to fetch OHLCV data for the specified crypto pair
            for start, end in intervals:
                ohlcv_url = f"https://rest.coinapi.io/v1/ohlcv/{symbol_id}/history"
                params = {
                    'period_id': f'{timestamp}',
                    'time_start': start.strftime('%Y-%m-%dT00:00:00'),
                    'time_end': end.strftime('%Y-%m-%dT00:00:00'),
                    'limit': 30000
                }

                response = requests.get(ohlcv_url, headers=self.headers, params=params)

                if response.status_code == 200:
                    ohlcv_data = response.json()
                    data = {
                        'Date': [entry["time_period_start"][:10] for entry in ohlcv_data],
                        'Open': [entry["price_open"] for entry in ohlcv_data],
                        'High': [entry["price_high"] for entry in ohlcv_data],
                        'Low': [entry["price_low"] for entry in ohlcv_data],
                        'Close': [entry["price_close"] for entry in ohlcv_data]
                    }
                    interval_df = pd.DataFrame(data)

                    # Append current interval data to the main DataFrame
                    all_data = pd.concat([all_data, interval_df], ignore_index=True)

                else:
                    print(f"Error fetching OHLCV data for {start} to {end}:", response.status_code, response.text)

            return all_data
        else:
            print("Error fetching pairs:", response.status_code, response.text)
            return None
        lg.info("fetch_crypto_data method executed successfully")
