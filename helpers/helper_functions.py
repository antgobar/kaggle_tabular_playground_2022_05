

import pandas as pd
from typing import Tuple
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pickle import dump, load

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
        to_join = to_join.reset_index(drop=True).join(encoded_df.reset_index(drop=True))
    return to_join

def compute_interations(df: pd.DataFrame):
    df['i_02_21'] = (df.f_21 + df.f_02 > 5.2).astype(int) - (df.f_21 + df.f_02 < -5.3).astype(int)
    df['i_05_22'] = (df.f_22 + df.f_05 > 5.1).astype(int) - (df.f_22 + df.f_05 < -5.4).astype(int)
    i_00_01_26 = df.f_00 + df.f_01 + df.f_26
    df['i_00_01_26'] = (i_00_01_26 > 5.0).astype(int) - (i_00_01_26 < -5.0).astype(int)
    return df 

def get_prepared_data(X):
    assert 'target' not in X.columns
    X = compute_interations(X)
    types = X.dtypes
    num_cols = list(types[types != object].index)
    cat_cols = list(types[types == object].index)
    scaler_cols = list(types[types == float].index)
    X_num = X[num_cols]
    X_cat = X[cat_cols]
    X_cat_encoded = encode_string_value(X_cat)
    df = X_num.join(X_cat_encoded)
    return df, scaler_cols 

def get_scaled_data(X, is_test = False):
    """ dumps scaler to the model folder, only scales needed cols """
   
    encod_data, scaler_cols = get_prepared_data(X)
    encod_data_copy = encod_data.copy()
    features = encod_data_copy[scaler_cols]
    if not is_test:
        scaler = StandardScaler()
        scaler.fit(features.values)
        dump(scaler, open('models/standard_scaler.pkl', 'wb'))
    else: 
        scaler = load(open('models/standard_scaler.pkl', 'rb'))
        
    normal_data = scaler.transform(features.values)
    encod_data_copy[scaler_cols] = normal_data
    return encod_data_copy

