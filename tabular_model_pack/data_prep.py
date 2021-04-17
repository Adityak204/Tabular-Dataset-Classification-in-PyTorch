import pandas as pd
import numpy as np
from sklearn import preprocessing


class DataPrep:
    def __init__(self, data, categorical_var_list):
        self.data = data
        self.categorical_var_list = categorical_var_list

    def missing_value_impute(self):
        for col in self.data.columns:
            if col in self.categorical_var_list:
                temp_col = self.data[col].fillna('NONE').astype(str).values
                self.data.loc[:, col] = temp_col

    def label_encoding(self):
        label_encoder = {}
        for col in self.data.columns:
            if col in self.categorical_var_list:
                label_encoder[col] = preprocessing.LabelEncoder()
                self.data[col] = label_encoder[col].fit_transform(self.data[col])

    def run_preprocessing(self, treat_na=None, label_encode=None):
        if treat_na:
            self.missing_value_impute()
        if label_encode:
            self.label_encoding()
        return self.data


if __name__ == '__main__':
    dt = pd.DataFrame({'category': ['a', 'b', 'c', 'a', 'a', 'c'],
                       'class': ['I', 'IV', None, 'I', 'I', 'V']
                       })
    test1 = DataPrep(data=dt, categorical_var_list=dt.columns)
    t1_dt = test1.run_preprocessing(treat_na=True)
    print(t1_dt['class'].value_counts())
    test2 = DataPrep(data=dt, categorical_var_list=dt.columns)
    t2_dt = test1.run_preprocessing(treat_na=True, label_encode=True)
    print(t2_dt['class'].value_counts())
