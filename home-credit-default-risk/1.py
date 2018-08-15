
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


def handleXNA(df):
    df.replace('XNA', np.nan, inplace=True)
    df.replace('XAP', np.nan, inplace=True)
    return df


# In[85]:


df_train = pd.DataFrame()
df_test = pd.DataFrame()
df_bb = pd.DataFrame()
def load_data():
    global df_train
    df_train = pd.read_csv('../Data/application_train.csv')
    df_train = handleXNA(df_train)
    global df_test
    df_test = pd.read_csv('../Data/application_test.csv')
    df_test = handleXNA(df_test)
    global df_bb
    df_bb = pd.read_csv('../Data/bureau_balance.csv')
    df_bb = handleXNA(df_bb)


# In[5]:


def label_encoding(df, column):
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(df[column].unique())
    list(le.classes_)
    df[column] = le.transform(df[column])
    return df


# In[6]:


def fillNa(df):
    for col in df.columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].value_counts().idxmax(), inplace=True)


# # EDA

# In[86]:


load_data()


# In[89]:


df_bb.columns


# In[88]:


df_train.describe()


# In[8]:


print(', '.join(list(df_train.columns)))
print(len(df_train))


# In[9]:


y_train = df_train['TARGET']
df_train.drop(['TARGET'], axis=1, inplace=True)
train_size = len(df_train)
test_size = len(df_test)
df = pd.concat([df_train, df_test], axis=0)


# In[10]:


print(df['DAYS_BIRTH'].describe())
df['DAYS_BIRTH'] = df['DAYS_BIRTH'] / (-365)
print(df['DAYS_BIRTH'].describe())


# In[11]:


print(df['DAYS_EMPLOYED'].describe())
df['DAYS_EMPLOYED_ERR'] = (df['DAYS_EMPLOYED'] == df['DAYS_EMPLOYED'].max())
df['DAYS_EMPLOYED'].replace({df['DAYS_EMPLOYED'].max(): np.nan}, inplace=True)
print(df['DAYS_EMPLOYED'].describe())


# # Feature Engineering

# In[22]:


poly_features = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

from sklearn.preprocessing import Imputer
imputer = Imputer(strategy='median')

poly_features = imputer.fit_transform(poly_features)


# In[23]:


from sklearn.preprocessing import PolynomialFeatures

poly_transformer = PolynomialFeatures(degree=3)
poly_transformer.fit(poly_features)
poly_features = poly_transformer.transform(poly_features)
print(poly_transformer.get_feature_names)


# In[26]:


poly_features_df = pd.DataFrame(poly_features, columns=poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']))
poly_features_df['TARGET'] = y_train
poly_corrs = poly_features_df.corr()['TARGET'].sort_values()
print(poly_corrs.head(10))
print(poly_corrs.tail(10))


# In[35]:


poly_features_df_train = poly_features_df[:train_size]
poly_features_df_test = poly_features_df[train_size:train_size+test_size]


# In[36]:


# Merging poly_features_df
if 'TARGET' in poly_features_df_train.columns:
    poly_features_df_train.drop(['TARGET'], axis=1, inplace=True)
poly_features_df_train['SK_ID_CURR'] = df_train['SK_ID_CURR']
poly_features_df_test['SK_ID_CURR'] = df_test['SK_ID_CURR']

# poly_df_train = poly_features_df[:train_size]
# poly_df_test = poly_features_df[train_size:train_size:test_size]

poly_df_train = df_train.merge(poly_features_df_train, on='SK_ID_CURR', how='left')
poly_df_test = df_test.merge(poly_features_df_test, on='SK_ID_CURR', how='left')

poly_df = pd.concat([poly_df_train, poly_df_test], axis=0)


# ## Fill na values

# In[37]:


def fillNa(df):
    for col in df.columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].value_counts().idxmax(), inplace=True)


# In[38]:


def one_hot_encode(df, column):
    df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
    df.drop([column], axis=1, inplace=True)
    return df


# In[39]:


def label_encoding(df, column):
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(df[column].unique())
    list(le.classes_)
    df[column] = le.transform(df[column])
    return df


# In[40]:


def prepare(df):
    fillNa(df)
    for col in df.columns:
        if ("AMT_REQ_CREDIT" not in col) and ("DEF_30_CNT_SOCIAL_CIR" not in col) and ("DEF_60_CNT_SOCIAL_CIR" not in col):
            if len(df[col].unique()) > 2 and len(df[col].unique()) < 9:
                print(col, len(df[col].unique()))
                df = label_encoding(df, col)
                df = one_hot_encode(df, col)
    for col in df.columns:
        if df[col].dtype == object:
            print(col, len(df[col].unique()))
            df = label_encoding(df, col)
            if len(df[col].unique()) > 2:
                df = one_hot_encode(df, col)
    print(', '.join(list(df.columns)))
    return df


# In[41]:


poly_df['REGION_RATING_CLIENT_W_CITY'].replace(-1, poly_df['REGION_RATING_CLIENT_W_CITY'].value_counts().idxmax(), inplace=True)
poly_df = prepare(poly_df)


# In[42]:


df_train = poly_df[0:train_size]
df_test = poly_df[train_size:test_size+train_size]


# In[30]:


temp_df = df_train
temp_df['TARGET'] = y_train
corr = temp_df.corr()['TARGET'].sort_values()
print(corr.tail(10))
print('-----------------------')
print(corr.head(10))


# In[43]:


df_train.drop(['SK_ID_CURR'], axis=1, inplace=True)
b_id = df_test['SK_ID_CURR']
df_test.drop(['SK_ID_CURR'], axis=1, inplace=True)


# In[44]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train, y_train, test_size=0.2)


# # XGBoost

# In[45]:


import xgboost as xgb


# In[46]:


import os
os.environ['http_proxy']="http_proxy"
os.environ['https_proxy']="https_proxy"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='3'


# In[47]:


num_class = len(np.unique(y_train))
print(num_class)


# In[177]:


# dtrain = xgb.DMatrix(X_train, label=y_train)
# dtest = xgb.DMatrix(X_test, label=y_test)


# In[179]:


# print(dtrain.size())


# In[51]:


params = {'max_depth': 12, 'n_estimators': 300, 'tree_method': 'gpu_hist', 'n_gpus':1 , 'backend':'h2o4gpu',
              'random_state': 123 , 'n_jobs': -1, 'predictor': 'gpu_predictor', 'verbose':1, 'eval_metric': 'logloss'}


# In[52]:


model = xgb.XGBClassifier(**params)


# In[53]:


model.fit(X_train, y_train, early_stopping_rounds=100, eval_set=[(X_test, y_test)])


# In[54]:


result = model.predict_proba(df_test)


# In[55]:


result


# In[56]:


result[:, -1]


# In[57]:


# col = [building_id, damage_grade]
subm = pd.DataFrame()
subm['SK_ID_CURR'] = b_id
ans = pd.DataFrame(data=result[:, -1], columns=['TARGET'])
subm = pd.concat([subm, ans], axis=1)
subm.to_csv('../Submissions/2.csv', index=False)
# subm.head(100)

