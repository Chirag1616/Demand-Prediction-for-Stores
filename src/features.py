import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

def feature_engineering(df, target_col="Weekly_Sales", lags=(1,2,3)):
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df['Year'] = df['Date'].dt.year
        df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
        df['Month'] = df['Date'].dt.month
    else:
        df['Year'] = df['Week'] = df['Month'] = 0

    df['week_sin'] = np.sin(2 * np.pi * df['Week'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['Week'] / 52)
    df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    num_cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df[c] = df[c].fillna(df[c].median())

    if 'CPI' in df.columns and 'Fuel_Price' in df.columns:
        df['CPI_Fuel'] = df['CPI'] * df['Fuel_Price']
    if 'Temperature' in df.columns and 'Unemployment' in df.columns:
        df['Temp_Unemployment'] = df['Temperature'] * df['Unemployment']

    group_col = 'Store' if 'Store' in df.columns else None
    if group_col:
        for lag in lags:
            df[f'sales_lag_{lag}'] = df.groupby(group_col)[target_col].shift(lag)
        df['sales_roll_mean_3'] = (
            df.groupby(group_col)[target_col]
              .shift(1).rolling(window=3).mean().reset_index(level=0, drop=True)
        )
    else:
        for lag in lags:
            df[f'sales_lag_{lag}'] = df[target_col].shift(lag)
        df['sales_roll_mean_3'] = df[target_col].shift(1).rolling(window=3).mean()

    if 'IsHoliday' in df.columns:
        df['IsHoliday'] = df['IsHoliday'].astype(int)

    df = df.dropna().reset_index(drop=True)
    return df

def prepare_features(df, target_col="Weekly_Sales"):
    candidate_feats = [
        'Temperature','Fuel_Price','CPI','Unemployment',
        'CPI_Fuel','Temp_Unemployment',
        'Year','Week','Month',
        'week_sin','week_cos','month_sin','month_cos',
        'sales_lag_1','sales_lag_2','sales_lag_3','sales_roll_mean_3',
        'IsHoliday'
    ]
    features = [c for c in candidate_feats if c in df.columns]

    for c in ['Store','Dept']:
        if c in df.columns:
            le = LabelEncoder()
            df[c+'_enc'] = le.fit_transform(df[c].astype(str))
            features.append(c+'_enc')

    X = df[features].fillna(df[features].median())
    y = df[target_col].astype(float)
    return X, y, df
