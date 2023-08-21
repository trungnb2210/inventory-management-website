import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX

stock = pd.read_csv('C:/Users/Admin/Downloads/stock.csv')
train = pd.read_csv('D:/Funnyland/excel/traindata.csv')
test = pd.read_csv('D:/Funnyland/excel/testdata.csv')
holiday = pd.read_csv('D:/Funnyland/excel/trainhol.csv')
holiday_test = pd.read_csv('D:/Funnyland/excel/testhol.csv')

train['date'] = pd.to_datetime(train.date, format = '%d/%m/%Y')
test['date'] = pd.to_datetime(test.date, format = '%d/%m/%Y')
holiday['date'] = pd.to_datetime(holiday.date, format = '%d/%m/%Y')
holiday_test['date'] = pd.to_datetime(holiday_test.date, format = '%d/%m/%Y')

train['bill'].fillna(method='ffill', inplace=True)
train['date'].fillna(method='ffill', inplace=True)
test['bill'].fillna(method='ffill', inplace=True)
test['date'].fillna(method='ffill', inplace=True)

pivot_daily_train = train.pivot_table(index='date', columns='code', values='quantity', fill_value = 0, aggfunc='sum')


all_dates = pd.date_range(start=pivot_daily_train.index.min(),
                          end=pivot_daily_train.index.max(),
                          freq='D')


pivot_daily_train = pivot_daily_train.reindex(all_dates, fill_value=0)
pivot_daily_train.reset_index(inplace=True)

code_set = set(pivot_daily_train.columns) - {'index'}

test_filtered = test[test['code'].isin(code_set)]

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

order = (2, 1, 2)
seasonal_order = (0, 1, 1, 7)

sarima_model = SARIMAX(train_aggregated['sales'], exog=exog_train, order=order, seasonal_order=seasonal_order)
sarima_result = sarima_model.fit()

forecast_period = 90
sarima_forecast = sarima_result.get_forecast(steps=forecast_period, exog=exog_test)

predicted_values = sarima_forecast.predicted_mean
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
plt.plot(train_aggregated.index, train_aggregated['sales'], label='Training Data')

# Plot actual test data
plt.plot(test_aggregated.index, test_aggregated['sales'], label='Actual Test Data', color='orange')

# Plot forecasted test data
plt.plot(compare_data.index, compare_data['predicted_sales'], label='Forecasted Test Data (Predicted)', color='red')
plt.fill_between(compare_data.index, compare_data['lower sales'], compare_data['upper sales'], color='pink', alpha=0.2)

# Plot holidays for training data
train_holiday_dates = train_merged_data[train_merged_data['holiday'] == 1]['index']
plt.scatter(train_holiday_dates, [max(train_aggregated['sales'])] * len(train_holiday_dates), color='green', marker='x', label='Holiday (Train)')

# Plot holidays for test data
test_holiday_dates = test_merged_data[test_merged_data['holiday'] == 1]['index']
plt.scatter(test_holiday_dates, [max(train_aggregated['sales'])] * len(test_holiday_dates), color='blue', marker='x', label='Holiday (Test)')

plt.title('Training and Forecasted Test Data with Holidays')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
