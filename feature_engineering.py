import pandas as pd

train_data = pd.read_csv("data/train_data.csv")
test_data = pd.read_csv("data/test_data.csv")

print(train_data.head(1))

train_dtObj = pd.DatetimeIndex(train_data['Date'])
train_data['year'] = train_dtObj.year
train_data['month'] = train_dtObj.month
train_data['day'] = train_dtObj.day


print(train_data.isnull().sum())
