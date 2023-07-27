import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Read the .csv file and parse the date
stock_data = pd.read_csv('C:/Users/Admin/Downloads/stock.csv')
SGsorted = pd.read_csv('D:/Funnyland/excel/sortedSG.csv')

SGsorted['date'] = pd.to_datetime(SGsorted['date'], dayfirst=True)  # Convert 'date' column to datetime type
SGsorted['bill'].fillna(method='ffill', inplace=True)
SGsorted['date'].fillna(method='ffill', inplace=True)

# Find the minimum date in the dataset
min_date = SGsorted['date'].min()

# Convert 'date' to numerical values (number of days since the minimum date)
SGsorted['date_numeric'] = (SGsorted['date'] - min_date).dt.days

# Convert the 'date_numeric' column to int64
SGsorted['date_numeric'] = SGsorted['date_numeric'].astype('int64')

item_codes_set = set(stock_data['code'])

# Initialize an empty list to store the filtered data
filtered_data = []

# Iterate through each row in the data and filter based on item codes
for _, row in SGsorted.iterrows():
    if row['code'] in item_codes_set:
        filtered_data.append(row)

# Create a new DataFrame from the filtered_data list
filtered_df = pd.DataFrame(filtered_data)

# Group by 3-day intervals and code, sum the quantity to get the total quantity sold for the selected items
daily_sales_filtered = filtered_df.groupby([pd.Grouper(key='date', freq='3D'), 'code'])['quantity'].sum().unstack(fill_value=0)

# Perform ARIMA modeling and forecasting for each item code
forecast_data = {}
for item_code in daily_sales_filtered.columns:
    item_sales = daily_sales_filtered[item_code]

    # Perform ARIMA modeling and forecasting
    model = ARIMA(item_sales, order=(10, 2, 2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=10)  # Forecast 10 intervals (30 days for 3-day intervals)

    # Apply a floor value to the forecasted values
    floor_value = 0  # Set your desired floor value here
    forecast = forecast.clip(lower=floor_value)

    # Round the forecasted values to the nearest integers
    forecast = forecast.round()

    # Store the forecasted values for the item code
    forecast_data[item_code] = forecast

# Plot the historical and forecasted sales for each item code one at a time
for item_code in daily_sales_filtered.columns:
    plt.figure(figsize=(12, 6))

    historical_data = daily_sales_filtered[item_code]
    forecast = forecast_data[item_code]
    
    # Create date range for the historical data
    historical_dates = daily_sales_filtered.index.get_level_values(0)
    plt.plot(historical_dates, historical_data, label=f'Item Code {item_code} - Historical Data', color='blue')
    
    # Create date range for the forecasted values
    forecast_dates = pd.date_range(start=historical_dates.max() + pd.to_timedelta(3, 'D'), periods=len(forecast), freq='3D')
    plt.plot(forecast_dates, forecast, label=f'Item Code {item_code} - Forecast', color='orange')
    
    plt.xlabel('Date')
    plt.ylabel('Quantity Sold')
    plt.legend()
    plt.title(f'Item Code {item_code}')
    plt.show()
