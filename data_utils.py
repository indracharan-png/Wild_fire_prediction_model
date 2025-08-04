import pandas as pd
import os


def load_data():
    
    # Getting the file path
    file_path = os.path.join(os.getcwd(), 'data', 'CA_Weather_Fire_Dataset_1984-2025.csv')
    df = pd.read_csv(file_path)

    # print(df.shape)
    # return

    # Drop null valued samples if any exist
    if df.isnull().values.any():
        df = df.dropna(axis=0)

    # Converting string to datetime object and extracting year, month, and day
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')
    df["YEAR"] = df["DATE"].dt.year
    df["MONTH"] = df["DATE"].dt.month
    df["DAY"] = df["DATE"].dt.day
    df.drop("DATE", axis=1, inplace=True)

    # Setting up the features and target
    features = [x for x in df.columns if x != 'FIRE_START_DAY']
    target = 'FIRE_START_DAY'

    X = df[features]
    y = df[target].astype('int')
    # print(X.shape)

    # One-hot encoding categorical features
    X = pd.get_dummies(X, drop_first=True)
    X = X.astype('float32')

    return X, y
  




if __name__ == "__main__":
    load_data()