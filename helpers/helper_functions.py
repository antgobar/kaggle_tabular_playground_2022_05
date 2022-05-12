
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import OneHotEncoder

def load_data(dir: str or None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if dir:
        return pd.read_csv(dir + '/train.csv', index_col = 'id'), pd.read_csv(dir + '/test.csv', index_col = 'id')
    else:
        return pd.read_csv('/train.csv', index_col = 'id'), pd.read_csv('/test.csv', index_col = 'id')

def encode_string_value(X_cat: pd.DataFrame):
    X_cat_split=X_cat['f_27'].str.split('', expand = True).drop([0, 11], axis=1)
    to_join = X_cat_split[[]]
    for i in range(10):
        series = pd.DataFrame(X_cat_split.iloc[:,i])
        enc = OneHotEncoder(categories='auto',)
        encoded = enc.fit_transform(series)
        col_names = [f'{i+1}_' + l for l in list(enc.categories_[0])]
        encoded_df = pd.DataFrame(encoded.toarray(), columns = col_names)
        to_join = to_join.join(encoded_df)
    return to_join

def get_prepared_data(X):
    assert 'target' not in X.columns
    types = X.dtypes
    num_cols = list(types[types != object].index)
    cat_cols = list(types[types == object].index)
    X_num = X[num_cols]
    X_cat = X[cat_cols]
    X_cat_encoded = encode_string_value(X_cat)
    df = X_num.join(X_cat_encoded)
    return df