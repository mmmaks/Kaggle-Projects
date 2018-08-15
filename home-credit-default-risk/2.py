
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


def handleXNA(df):
    df.replace('XNA', np.nan, inplace=True)
    df.replace('XAP', np.nan, inplace=True)
    return df


# In[5]:


df_train = pd.DataFrame()
df_test = pd.DataFrame()
df_b = pd.DataFrame()
df_bb = pd.DataFrame()
def load_data():
    global df_train
    df_train = pd.read_csv('../Data/application_train.csv')
    df_train = handleXNA(df_train)
    global df_test
    df_test = pd.read_csv('../Data/application_test.csv')
    df_test = handleXNA(df_test)
    global df_b
    df_b = pd.read_csv('../Data/bureau.csv')
    df_b = handleXNA(df_b)
    global df_bb
    df_bb = pd.read_csv('../Data/bureau_balance.csv')
    df_bb = handleXNA(df_bb)


# In[6]:


def one_hot_encode(df, column):
    df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
    df.drop([column], axis=1, inplace=True)
    return df


# In[7]:


def label_encoding(df, column):
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(df[column].unique())
    list(le.classes_)
    df[column] = le.transform(df[column])
    return df


# In[17]:


def fillNa(df):
    for col in df.columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].value_counts().idxmax(), inplace=True)


# In[10]:


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


# # EDA

# In[11]:


load_data()
print(len(df_train.columns) - len(df_test.columns))
y_train = df_train[['SK_ID_CURR', 'TARGET']]
df_train.drop(['TARGET'], axis=1, inplace=True)


# In[12]:


df_train['is_train'] = 1
df_train['is_test'] = 0
df_test['is_test'] = 1
df_test['is_train'] = 0


# In[18]:


df = pd.concat([df_train, df_test], axis=0)
df['DAYS_BIRTH'] = df_train['DAYS_BIRTH'] / (-365)
df['REGION_RATING_CLIENT_W_CITY'].replace(-1, df['REGION_RATING_CLIENT_W_CITY'].value_counts().idxmax(), inplace=True)
df = prepare(df)


# # Bureau Table

# In[19]:


df_bb.STATUS.unique()


# In[20]:


df_b_bb = df_b.merge(df_bb, on='SK_ID_BUREAU', how='left')


# In[21]:


fillNa(df_b_bb)
df_b_bb.STATUS.unique()


# In[22]:


df_b_CAT = df_b_bb[['SK_ID_CURR', 'CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE', 'STATUS']]
df_b_CAT.head()


# In[23]:


cat_feature = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE', 'STATUS']
for feature in cat_feature:
    df_b_CAT = label_encoding(df_b_CAT, feature)
    df_b_CAT = one_hot_encode(df_b_CAT, feature)


# In[24]:


df_b_CAT.head()


# In[25]:


def aggregate(df, col):
    df = df.groupby(col).agg(['min', 'max', 'var', 'sum'])
    df = df.reset_index()
    df.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in df.columns.tolist()])
    for x in df.columns:
        if col in x:
            df.rename(columns={x: col}, inplace=True)
    return df


# In[30]:


to_merge = df_b_CAT.groupby('SK_ID_CURR').sum()


# In[31]:


print(len(to_merge))
print(len(df))


# In[34]:


to_merge = to_merge.reset_index()


# In[35]:


to_merge.head()


# In[36]:


ct = 0
for id in df_test['SK_ID_CURR']:
    if id in to_merge['SK_ID_CURR']:
        ct += 1
print(ct)


# In[37]:


df = df.merge(to_merge, on='SK_ID_CURR', how='left')


# In[38]:


df.head()


# In[39]:


len(df.columns)


# In[40]:


df_b_bb.head()


# In[41]:


numeric_features = [col for col in df_b_bb.columns if col not in ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE', 'STATUS']]


# In[42]:


num_df = df_b_bb[numeric_features]


# In[43]:


num_df.head()


# In[44]:


num_df.drop(['SK_ID_BUREAU'], axis=1, inplace=True)


# In[46]:


def fillNaNumerical(df):
    for col in df.columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)


# In[47]:


fillNaNumerical(num_df)


# In[48]:


num_to_merge = aggregate(num_df, 'SK_ID_CURR')


# In[49]:


# num_to_merge = num_to_merge.reset_index()


# In[50]:


num_to_merge.head()


# In[51]:


df = df.merge(num_to_merge, on='SK_ID_CURR', how='left')


# In[52]:


df['SK_ID_CURR'].head()


# In[56]:


for col in df.columns:
    if df[col].dtype == object:
        print(col)


# In[53]:


df.to_csv('../Merged_data/df_train_test_b_bb.csv', index=False)


# In[54]:


y_train.head()


# In[55]:


y_train.to_csv('../Merged_data/labels.csv', index=False)

