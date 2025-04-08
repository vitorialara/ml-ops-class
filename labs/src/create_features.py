import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectPercentile, f_classif
import pickle
import yaml

# load params
params = yaml.safe_load(open("params.yaml"))["features"]
data_path = params["data_path"]
use_scaler = params["scale_numeric"]

# read data
train = pd.read_csv(f'{data_path}/adult.data', header=None)
test = pd.read_csv(f'{data_path}/adult.test', header=None)

cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

train.columns = cols
test.columns = cols


train['income'] = train['income'].str.strip()
test['income'] = test['income'].str.strip()


X_train = train.drop('income', axis=1)
y_train = train['income']
X_test = test.drop('income', axis=1)
y_test = test['income']


num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
cat_cols = ['workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'native-country']


# preprocessing pipeline with optional scaling
num_pipe = StandardScaler() if use_scaler else 'passthrough'
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipe, num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])


# feature selection with param
selector = SelectPercentile(f_classif, percentile=params["percentile"])


pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('selector', selector)
])


X_train_processed = pipe.fit_transform(X_train, y_train)
X_test_processed = pipe.transform(X_test)


with open(f'{data_path}/pipeline.pkl', 'wb') as f:
    pickle.dump(pipe, f)


pd.DataFrame(X_train_processed).to_csv(f'{data_path}/processed_train_data.csv', index=False)
pd.DataFrame(X_test_processed).to_csv(f'{data_path}/processed_test_data.csv', index=False)
