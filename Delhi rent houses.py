""" first number is 25th percentile among ~4000 houses, the second number is 75th percentile among ~4000 houses
latitude 28.45732-28.60338
longitude 77.13890-77.22882
numBathrooms 2-4
numBalconies 0-2
isNegotiable 0
verificationDate 20-365
Status 1-2
Size_m2 120-548
BHK 1-1
rooms_num 3-4
SecurityDeposit_euro 0-10829
"""

#for user:
latitude = 28.45732
longitude = 77.13890
numBathrooms = 2
numBalconies = 2
isNegotiable = 0
verificationDate = 20
Status = 1
Size_m2 = 120
BHK = 1
rooms_num = 3
SecurityDeposit_euro = 0

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor


def remove_outliers(df):
    df_clean = df.copy()
    for col in df.select_dtypes(include='number').columns:
        quantile1 = 0.25
        Q1 = df[col].quantile(quantile1)
        Q3 = df[col].quantile(1 - quantile1)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean


def convert_to_days(time_str):
    time_part = time_str.replace('Posted ', '').replace(' ago', '')
    parts = time_part.split()

    if parts[0] == 'a' or parts[0] == 'an':
        value = 1
        unit = parts[1]
    else:
        value = int(parts[0])
        unit = parts[1]

    if unit in ['minute', 'minutes']:
        return value / (24 * 60)
    elif unit in ['hour', 'hours']:
        return value / 24
    elif unit in ['day', 'days']:
        return value
    elif unit in ['month', 'months']:
        return value * 30
    elif unit in ['year', 'years']:
        return value * 365
    else:
        return np.nan


# -----------------------------------------------------
pd.set_option('display.max_rows', 999999999)
pd.set_option('display.max_columns', 999999999)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)

data = pd.read_csv('https://raw.githubusercontent.com/Lizar1/Delhi_price_prediction/refs/heads/main/Indian_housing_Delhi_data.csv')

data['Size_split'] = data['house_size'].str.split().str[0]
data['Size_split'] = data['Size_split'].str.replace(',', '', regex=False).astype(int)

data.rename(columns={'Size_split': 'Size_ft²'}, inplace=True)

data.drop(columns=['currency', 'house_size', 'priceSqFt'], inplace=True)

data['Size_m2'] = (data['Size_ft²'] / 10.76).round(1)
data.drop(columns=['Size_ft²'], inplace=True)

data['price_euro'] = (data['price'] / 93.45).round(1)
data.drop(columns=['price'], inplace=True)

data['numBalconies'] = data['numBalconies'].fillna(0).astype(int)

data['SecurityDeposit'] = data['SecurityDeposit'].str.replace(',', '', regex=False)
data['SecurityDeposit'] = data['SecurityDeposit'].str.strip().replace('No Deposit', '0', regex=False).astype(int)

le = LabelEncoder()
data['Status'] = le.fit_transform(data['Status'])
data['city'] = le.fit_transform(data['city'])

data['BHK_type'] = data['house_type'].str.split().str[1]
data['BHK'] = data['BHK_type'].map({'RK': 0, 'BHK': 1})
data['rooms_num'] = data['house_type'].str.split().str[0].astype(int)

data.drop(columns=['BHK_type'], inplace=True)

data['property_type'] = data['house_type'].str.split().apply(lambda x: ' '.join(x[2:4]))
data.drop(columns=['house_type'], inplace=True)

data['verificationDate'] = data['verificationDate'].apply(convert_to_days)
data.drop(columns=['description'], inplace=True)

data['isNegotiable'] = data['isNegotiable'].fillna(0)
data['isNegotiable'] = data['isNegotiable'].replace('Negotiable', 1).astype(int)

data = data.dropna(subset=["numBathrooms"])

data['SecurityDeposit_euro'] = (data['SecurityDeposit'] / 93.45).round(1)
data.drop(columns=['SecurityDeposit'], inplace=True)

data.drop(columns=['property_type'], inplace=True)
data.drop(columns=['location'], inplace=True)
data.drop(columns=['city'], inplace=True)

data = remove_outliers(data)
#print(data.describe().round(5))
#print(data.head(10))
# -------------------------------------------------------------
X = data.drop(columns=["price_euro"])
y = data["price_euro"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# KNN
knn = KNeighborsRegressor(n_neighbors=4)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
rmse_knn = mean_squared_error(y_test, y_pred_knn) ** 0.5
print("KNN RMSE:", rmse_knn)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
rmse_lr = mean_squared_error(y_test, y_pred_lr) ** 0.5
print("Linear Regression RMSE:", rmse_lr)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rmse_rf = mean_squared_error(y_test, y_pred_rf) ** 0.5
print("Random Forest RMSE:", rmse_rf)

# -----------------------------------µ
tolerance = 0.2

within_20_percent_knn = ((y_pred_knn >= (1 - tolerance) * y_test) &
                         (y_pred_knn <= (1 + tolerance) * y_test)).mean()
print("Доля предсказаний KNN в пределах +-20%:", within_20_percent_knn)

within_20_percent_lr = ((y_pred_lr >= (1 - tolerance) * y_test) &
                        (y_pred_lr <= (1 + tolerance) * y_test)).mean()
print("Доля предсказаний LR в пределах +-20%:", within_20_percent_lr)

within_20_percent_rf = ((y_pred_rf >= (1 - tolerance) * y_test) &
                        (y_pred_rf <= (1 + tolerance) * y_test)).mean()
print("Доля предсказаний RF в пределах +-20%:", within_20_percent_rf)


def predict_priceKNN(latitude, longitude, numBathrooms, numBalconies, isNegotiable, verificationDate,
                     Status, Size_m2, BHK, rooms_num, SecurityDeposit_euro):
    X_new = pd.DataFrame({
        "latitude": [latitude],
        "longitude": [longitude],
        "numBathrooms": [numBathrooms],
        "numBalconies": [numBalconies],
        "isNegotiable": [isNegotiable],
        "verificationDate": [verificationDate],
        "Status": [Status],
        "Size_m2": [Size_m2],
        "BHK": [BHK],
        "rooms_num": [rooms_num],
        "SecurityDeposit_euro": [SecurityDeposit_euro]
    })
    X_new_scaled = scaler.transform(X_new)
    price = knn.predict(X_new_scaled)
    return price[0]


def predict_priceLR(latitude, longitude, numBathrooms, numBalconies, isNegotiable, verificationDate,
                    Status, Size_m2, BHK, rooms_num, SecurityDeposit_euro):
    X_new = pd.DataFrame({
        "latitude": [latitude],
        "longitude": [longitude],
        "numBathrooms": [numBathrooms],
        "numBalconies": [numBalconies],
        "isNegotiable": [isNegotiable],
        "verificationDate": [verificationDate],
        "Status": [Status],
        "Size_m2": [Size_m2],
        "BHK": [BHK],
        "rooms_num": [rooms_num],
        "SecurityDeposit_euro": [SecurityDeposit_euro]
    })
    X_new_scaled = scaler.transform(X_new)
    price = lr.predict(X_new_scaled)
    return price[0]


def predict_priceRF(latitude, longitude, numBathrooms, numBalconies, isNegotiable, verificationDate,
                    Status, Size_m2, BHK, rooms_num, SecurityDeposit_euro):
    X_new = pd.DataFrame({
        "latitude": [latitude],
        "longitude": [longitude],
        "numBathrooms": [numBathrooms],
        "numBalconies": [numBalconies],
        "isNegotiable": [isNegotiable],
        "verificationDate": [verificationDate],
        "Status": [Status],
        "Size_m2": [Size_m2],
        "BHK": [BHK],
        "rooms_num": [rooms_num],
        "SecurityDeposit_euro": [SecurityDeposit_euro]
    })

    X_new_scaled = scaler.transform(X_new)
    price = rf.predict(X_new_scaled)
    return price[0]

print("KNN prediction:", round(
    predict_priceKNN(latitude, longitude, numBathrooms, numBalconies, isNegotiable, verificationDate, Status, Size_m2,
                     BHK, rooms_num, SecurityDeposit_euro), 2))
print("LR prediction:", round(
    predict_priceLR(latitude, longitude, numBathrooms, numBalconies, isNegotiable, verificationDate, Status, Size_m2,
                    BHK, rooms_num, SecurityDeposit_euro), 2))
print("RF prediction:", round(
    predict_priceRF(latitude, longitude, numBathrooms, numBalconies, isNegotiable, verificationDate, Status, Size_m2,
                    BHK, rooms_num, SecurityDeposit_euro), 2))
