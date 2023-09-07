import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

data = pd.read_csv('D:/Funnyland/inventory-management-website/excel/DATAFORDISPLAY.csv')

items = ['SG 455', 'MAG-701005', 'MAG-701003', 'SG 423', 'DL32650', 'HP32285']

data['date'] = pd.to_datetime(data.date, format = '%d/%m/%Y')
data['date'].fillna(method='ffill', inplace=True)
data['bill'].fillna(method='ffill', inplace=True)

data.set_index('date', inplace=True)

# Create a single plot for all items
plt.figure(figsize=(10, 6))

for code in items:
    data_item = data[data['code'] == code]
    monthly_sales = data_item['quantity'].resample('M').sum()
    monthly_sales_df = monthly_sales.to_frame().reset_index()
    monthly_sales_df.columns = ['date', 'monthly_sales']
    plt.plot(monthly_sales.index, monthly_sales.values, label=f'Item {code}')


# Customize the plot
plt.title('Monthly Sales for Multiple Items')
plt.xlabel('Date')
plt.ylabel('Monthly Sales')
plt.grid(True)
plt.legend(loc='upper left')

# Show the plot with all items
plt.show()