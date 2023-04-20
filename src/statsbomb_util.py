import pandas as pd
import json


def normalize_dataframe(df):
    json_data = json.loads(df.to_json(orient='records'))
    norm_df = pd.json_normalize(json_data)
    return norm_df


def rename_cols(df):
    new_cols = [i.replace('.', '_') for i in df.columns]
    df.columns = new_cols
    df = df.rename(columns={
        'period': 'period_id',
        'id': 'event_id',
    })
    return df


def normalize_time_column(df):
    df['timestamp_shift'] = df['timestamp'].shift(1).fillna(df['timestamp'].iloc[0])
    df['timediff'] = (df['timestamp'] - df['timestamp_shift']) / 1000
    df['time'] = df['timediff'].cumsum()
    df['time'] = df['time'].round(2)

    return df
