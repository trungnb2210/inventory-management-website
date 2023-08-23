# instead of taking aggregate, take one particular item that has been selling through out 2021 and 2022 up to 2023.
# a long-term toy is SG 455. This toy has been sold across the firm over 2021 and 2022 so it will be take as an example for this algorithm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

stock = pd.read_csv('D:/Funnyland/inventory-management-website/excel/stock.csv')
train23 = pd.read_csv('D:/Funnyland/inventory-management-website/excel/2023data.csv')
train22 = pd.read_csv('D:/Funnyland/inventory-management-website/excel/2022data.csv')
train21 = pd.read_csv('D:/Funnyland/inventory-management-website/excel/2021data.csv')
holiday23 = pd.read_csv('D:/Funnyland/inventory-management-website/excel/2023hol.csv')
holiday22 = pd.read_csv('D:/Funnyland/inventory-management-website/excel/2022hol.csv')
holiday21 = pd.read_csv('D:/Funnyland/inventory-management-website/excel/2021hol.csv')
test = pd.read_csv('D:/Funnyland/inventory-management-website/excel/2023Jultestdata.csv')
holiday_test = pd.read_csv('D:/Funnyland/inventory-management-website/excel/2023Jultesthol.csv')

item = 'SG 455'
stock = stock[stock['code'] == item][['code', 'stock']]
stock['stock'] = pd.to_numeric(stock['stock'], errors='coerce')

train = pd.concat([train21, train22, train23], ignore_index=True)
train['date'] = pd.to_datetime(train.date, format = '%d/%m/%Y')
test['date'] = pd.to_datetime(test.date, format = '%d/%m/%Y')
holiday = pd.concat([holiday21, holiday22, holiday23], ignore_index=True)
holiday['date'] = pd.to_datetime(holiday.date, format = '%d/%m/%Y')
holiday_test['date'] = pd.to_datetime(holiday_test.date, format = '%d/%m/%Y')

train['bill'].fillna(method='ffill', inplace=True)
train['date'].fillna(method='ffill', inplace=True)
test['bill'].fillna(method='ffill', inplace=True)
test['date'].fillna(method='ffill', inplace=True)

traindf = train[train['code'] == item]

pivot_daily_train = traindf.pivot_table(index='date', columns='code', values='quantity', fill_value = 0, aggfunc='sum')

all_dates = pd.date_range(start=pivot_daily_train.index.min(),
                          end=pivot_daily_train.index.max(),
                          freq='D')

pivot_daily_train = pivot_daily_train.reindex(all_dates, fill_value=0)
pivot_daily_train.reset_index(inplace=True)

code_set = set(pivot_daily_train.columns) - {'index'}

test_filtered = test[test['code'].isin(code_set)]
# end_date = '2023-07-01'
# test_filtered = test[(test['code'].isin(code_set)) & (test['date'] < end_date)]

pivot_daily_test = test_filtered.pivot_table(index='date', columns='code', values='quantity', fill_value = 0, aggfunc='sum')

all_dates_test= pd.date_range(start=pivot_daily_test.index.min(),
                              end=pivot_daily_test.index.max(),
                              freq='D')

pivot_daily_test = pivot_daily_test.reindex(all_dates_test, fill_value=0)
pivot_daily_test.reset_index(inplace=True)

holiday.fillna(0, inplace=True)
holiday['holiday'] = holiday['holiday'].astype(int)
holiday_test.fillna(0, inplace=True)
holiday_test['holiday'] = holiday_test['holiday'].astype(int)

# holiday_test = holiday_test[holiday_test['date'] < end_date]

train_sum_sales_col = pivot_daily_train.sum(axis=1)
test_sum_sales_col = pivot_daily_test.sum(axis=1)

train_sum_sales = pd.DataFrame({'index': pivot_daily_train['index'], 'sales': train_sum_sales_col})
test_sum_sales = pd.DataFrame({'index': pivot_daily_test['index'], 'sales': test_sum_sales_col})

train_merged_data = pd.merge(train_sum_sales, holiday, how='left', left_on='index', right_on='date')
train_merged_data.drop(columns=['date'], inplace=True)

exog_train = train_merged_data.loc[:, ['index', 'holiday']]
exog_train.set_index('index', inplace=True)

test_merged_data = pd.merge(test_sum_sales, holiday_test, how='left', left_on='index', right_on='date')
test_merged_data.drop(columns=['date'], inplace=True)

exog_test = test_merged_data.loc[:, ['index', 'holiday']]
exog_test.set_index('index', inplace=True)

train_aggregated = train_merged_data[['index','sales']].copy()
train_aggregated.set_index('index', inplace=True)

order = (2, 1, 1)
seasonal_order = (1, 1, 1, 7)

sarima_model = SARIMAX(train_aggregated['sales'], exog=exog_train, order=order, seasonal_order=seasonal_order)
sarima_result = sarima_model.fit()

forecast_period = 31
sarima_forecast = sarima_result.get_forecast(steps=forecast_period, exog=exog_test)

predicted_values = sarima_forecast.predicted_mean.round().astype(int)
#this is rounded due to the discrete nature of sales

confidence_intervals = sarima_forecast.conf_int()

predictions = pd.DataFrame({'index': predicted_values.index, 'predicted_sales': predicted_values})
predictions = pd.merge(predictions, confidence_intervals, left_index=True, right_index=True)

test_aggregated = test_merged_data[['index', 'sales']].copy()
test_aggregated.set_index('index', inplace=True)

# Merge the predicted values with the test data
compare_data = pd.merge(test_aggregated, predictions, left_index=True, right_index=True)

# Plot both training and test data along with holidays and forecasts
plt.figure(figsize=(12, 6))

# Plot training data
# plt.plot(train_aggregated.index, train_aggregated['sales'], label='Training Data')

# Plot actual test data
plt.plot(test_aggregated.index, test_aggregated['sales'], label='Actual Test Data', color='orange')

# Plot forecasted test data
plt.plot(compare_data.index, compare_data['predicted_sales'], label='Forecasted Test Data (Predicted)', color='red')
plt.fill_between(compare_data.index, compare_data['lower sales'], compare_data['upper sales'], color='pink', alpha=0.2)

# Plot holidays for training data
# train_holiday_dates = train_merged_data[train_merged_data['holiday'] == 1]['index']
# plt.scatter(train_holiday_dates, [max(train_aggregated['sales'])] * len(train_holiday_dates), color='green', marker='x', label='Holiday (Train)')

# Plot holidays for test data
test_holiday_dates = test_merged_data[test_merged_data['holiday'] == 1]['index']
plt.scatter(test_holiday_dates, [max(train_aggregated['sales'])] * len(test_holiday_dates), color='blue', marker='x', label='Holiday (Test)')

plt.title('Training and Forecasted Test Data with Holidays')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Initialize the stock level as of the first day of the forecast period
n = 100
stock_level, actual_stock_level = n, n
# stock_level, actual_stock_level = stock['stock'].values[0], stock['stock'].values[0] 
# Create lists to store stock levels and dates
actual_stock_levels = [actual_stock_level]
predicted_stock_levels = [stock_level]  # Initialize with the same initial stock level
dates = [compare_data.index[0]]
zero_stock_date = None

# Calculate the stock levels for each day of the forecast period
for i in range(1, forecast_period):
    stock_level -= compare_data['predicted_sales'].iloc[i - 1]  # Decrease stock based on predictions
    if stock_level <= 0 and zero_stock_date is None:
        zero_stock_date = compare_data.index[i]
    predicted_stock_levels.append(stock_level)  # Store predicted stock level
    actual_stock_level -= test_aggregated['sales'].iloc[i - 1]  # Calculate actual stock level (assuming it decreases by 1 each day)
    actual_stock_levels.append(actual_stock_level)  # Store actual stock level
    dates.append(compare_data.index[i])

# Plot actual and predicted stock levels over time
plt.figure(figsize=(12, 6))

# Plot actual stock level
plt.plot(dates, actual_stock_levels, label='Actual Stock', color='orange')

print(len(predicted_stock_levels))
# Plot predicted stock level
plt.plot(dates, predicted_stock_levels, label='Predicted Stock', color='red')

plt.axhline(y=0, color='gray', linestyle='--', label='Zero Stock Level')
if zero_stock_date:
    plt.axvline(x=zero_stock_date, color='blue', linestyle='--', label='Zero Stock Date')
    plt.annotate(
        'Zero Stock Date:' + str(zero_stock_date.strftime('%d-%m-%Y')),
        xy=(zero_stock_date, 0),
        xytext=(50, 30),  # Offset for text position
        textcoords='offset points',
        arrowprops=dict(arrowstyle='->', color='green')
    )
    
plt.title('Actual vs. Predicted Stock Levels')
plt.xlabel('Date')
plt.ylabel('Stock Quantity')
plt.legend()
plt.show()

#calculate the MEAN SQUARED ERROR of the model
mse = mean_squared_error(test_aggregated['sales'], compare_data['predicted_sales'])
print("The Mean Squared Error:" + str(mse))

#calculate the MEAN ABSOLUTE ERROR of the model
mae = mean_absolute_error(test_aggregated['sales'], compare_data['predicted_sales'])
print("The Mean Absolute Error:" + str(mae))