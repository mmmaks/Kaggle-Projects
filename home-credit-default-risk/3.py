
# coding: utf-8

# In[66]:


import pandas as pd
import numpy as np


# In[67]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[68]:


def handleXNA(df):
    df.replace('XNA', np.nan, inplace=True)
    df.replace('XAP', np.nan, inplace=True)
    return df


# In[69]:


df_credit = pd.DataFrame()
df_PC = pd.DataFrame()
df_PA = pd.DataFrame()
df_IP = pd.DataFrame()
def load_data():
    global df_credit
    df_credit = pd.read_csv('../Data/credit_card_balance.csv')
    df_credit = handleXNA(df_credit)
    global df_PC
    df_PC = pd.read_csv('../Data/POS_CASH_balance.csv')
    df_PC = handleXNA(df_PC)
    global df_PA
    df_PA = pd.read_csv('../Data/previous_application.csv')
    df_PA = handleXNA(df_PA)
    global df_IP
    df_IP = pd.read_csv('../Data/installments_payments.csv')
    df_IP = handleXNA(df_IP)


# In[70]:


def one_hot_encode(df, column):
    df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
    df.drop([column], axis=1, inplace=True)
    return df


# In[71]:


def label_encoding(df, column):
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(df[column].unique())
    list(le.classes_)
    df[column] = le.transform(df[column])
    return df


# In[123]:


def fillNa(df):
    for col in df.columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)


# In[73]:


def prepare(df):
    fillNa(df)
    for col in df.columns:
        if df[col].dtype == object:
            print(col, len(df[col].unique()))
            df = label_encoding(df, col)
            if len(df[col].unique()) > 2:
                df = one_hot_encode(df, col)
    print(', '.join(list(df.columns)))
    return df


# In[74]:


load_data()


# In[75]:


def categoricalData(df):
    cd = []
    for col in df.columns:
#         print(col, df[col].dtype)
        if df[col].dtype == object:
            cd.append((col, len(df[col].unique())))
    return cd


# In[76]:


print(categoricalData(df_PA))


# In[111]:


df_credit_prepared = prepare(df_credit)
df_PC_prepared = prepare(df_PC)
df_PA_prepared = prepare(df_PA)
df_IP_prepared = prepare(df_IP)


# In[106]:


temp_df = pd.DataFrame({'A': [1, 1, 2, 2], 'B': [1, 2, 3, 4], 'C': np.random.randn(4)})
temp_df = aggregate(temp_df, 'A')


# In[107]:


temp_df


# In[112]:


df_credit_prepared.drop(['SK_ID_PREV'], axis=1, inplace=True)
df_PC_prepared.drop(['SK_ID_PREV'], axis=1, inplace=True)
df_PA_prepared.drop(['SK_ID_PREV'], axis=1, inplace=True)
df_IP_prepared.drop(['SK_ID_PREV'], axis=1, inplace=True)


# In[114]:


def aggregate(df, col):
    df = df.groupby(col).agg(['min', 'max', 'var', 'sum'])
    df = df.reset_index()
    df.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in df.columns.tolist()])
    for x in df.columns:
        if col in x:
            df.rename(columns={x: col}, inplace=True)
    return df


# In[115]:


df_credit_prepared = aggregate(df_credit_prepared, 'SK_ID_CURR')
df_PC_prepared = aggregate(df_PC_prepared, 'SK_ID_CURR')
df_PA_prepared = aggregate(df_PA_prepared, 'SK_ID_CURR')
df_IP_prepared = aggregate(df_IP_prepared, 'SK_ID_CURR')


# # EDA

# In[118]:


df = df_IP_prepared
df = df.merge(df_PA_prepared, on='SK_ID_CURR', how='outer')
df = df.merge(df_PC_prepared, on='SK_ID_CURR', how='outer')
df = df.merge(df_credit_prepared, on='SK_ID_CURR', how='outer')


# In[119]:


cols = df.columns


# In[120]:


df = df.reset_index()


# In[57]:


# df['NAME_CONTRACT_STATUS_0_x']


# In[121]:


for col in df.columns:
    if "_x" in col:
        print(col, df[col].dtype)
        print(col[:-2] + "_y")


# In[122]:


temp_df = df
for col in df.columns:
    if "_x" in col:
        df[col] = (df[col] + df[col[:-2] + "_y"]) / 2
        df.drop([col[:-2] + "_y"], axis=1, inplace=True)


# In[124]:


fillNa(df)


# In[125]:


df.head()


# In[129]:


for col in df.columns:
    if df[col].dtype == object:
        print(col)


# In[126]:


df.drop(['index'], axis=1, inplace=True)


# In[127]:


df.to_csv('../Merged_data/C_PC_PA_IP.csv', index=False)

