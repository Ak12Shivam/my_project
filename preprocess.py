import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(data_file):
    liver_df = pd.read_csv(data_file)
    liver_df.dropna(inplace=True)
    liver_df['Gender'] = liver_df.Gender.map({'Female':2,'Male':1})
    x = liver_df.drop('Dataset',axis=1)
    y = liver_df['Dataset']
    x_std  = StandardScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_std,y,test_size=0.30,random_state=100)
    return x_train, x_test, y_train, y_test