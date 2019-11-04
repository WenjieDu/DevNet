# Credit fraud dataset is from https://www.kaggle.com/mlg-ulb/creditcardfraud,
# you could download it as you wish

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    data = pd.read_csv('dataset/creditcard.csv')
    data.drop(['Time', 'Class'], axis=1, inplace=True)
    data.rename(columns={'Class:class'}, inplace=True)

    train, test = train_test_split(data, test_size=0.2)

    train.to_csv('dataset/creditcard_train.csv', index=False)
    test.to_csv('dataset/creditcard_test.csv', index=False)

    test_x = test.drop('class', axis=1)
    test_y = test['class']
    np.save('dataset/creditcard_test_x.npy', test_x)
    np.save('dataset/creditcard_test_y.npy', test_y)
