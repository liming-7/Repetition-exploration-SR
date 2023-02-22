from datetime import datetime
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('dataset-train-diginetica/train-item-views.csv',
     sep=';',
     header= 0,
     index_col=False,
     usecols=[0, 2, 3])
    df = df.rename(columns={"sessionId": "user_id", "itemId": "item_id", "timeframe": "timestamp"})
    df.to_csv('diginetica-all.csv', index=False)