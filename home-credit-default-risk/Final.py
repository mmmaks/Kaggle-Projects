
# coding: utf-8

# In[35]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[36]:


df = pd.read_csv('../Merged_data/df_train_test_b_bb.csv')
df_merge = pd.read_csv('../Merged_data/C_PC_PA_IP.csv')
df_label = pd.read_csv('../Merged_data/labels.csv')


# In[37]:


df.head()


# In[38]:


df_merge.head()


# In[39]:


temp_df = df
df = df.merge(df_merge, on='SK_ID_CURR', how='left')


# In[66]:


for col in df.columns:
    if df[col].dtype == object:
        print(col)


# In[40]:


for col in df.columns:
    if "_y" in col:
        df[col] = (df[col] + df[col[:-2] + "_x"]) / 2
        df.drop([col[:-2] + "_x"], axis=1, inplace=True)


# In[41]:


df.head()


# In[42]:


df = df.reset_index()


# In[45]:


df.drop(['index'], axis=1, inplace=True)
df.head()


# In[43]:


df.describe()


# In[46]:


ignore_features = ['is_train', 'is_test']
relevant_features = [col for col in df.columns if col not in ignore_features]


# In[69]:


for col in df.columns:
    if df[col].dtype == object:
        print(col)


# In[83]:


trainX = df[df['is_train'] == 1][relevant_features]
testX = df[df['is_test'] == 1][relevant_features]


# In[84]:


for col in trainX.columns:
    trainX[col] = pd.to_numeric(trainX[col], errors='coerce')
for col in testX.columns:
    testX[col] = pd.to_numeric(testX[col], errors='coerce')


# In[85]:


# trainX.astype('float')
# testX.astype('float')


# In[86]:


testX.head()


# In[87]:


testX = testX.reset_index()
# testX.head()
# testX.drop(['index', 'level_0'], axis=1, inplace=True)
# trainX.drop(['index'], axis=1, inplace=True)


# In[90]:


testX.head()
testX.drop(['index'], axis=1, inplace=True)
# trainX.drop(['index'], axis=1, inplace=True)
testX.head()


# In[91]:


len(trainX.columns) == len(testX.columns)


# In[92]:


y_train = list(trainX.merge(df_label, on='SK_ID_CURR', how='left')['TARGET']).copy()
len(y_train)


# In[93]:


sk_id = testX['SK_ID_CURR']
trainX.drop(['SK_ID_CURR'], axis=1, inplace=True)
testX.drop(['SK_ID_CURR'], axis=1, inplace=True)


# In[65]:


# for col in trainX.columns:
#     print(col, trainX[col].dtype)


# In[96]:


def fillNa(df):
    for col in df.columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)


# In[95]:


cols_to_norm = []
for col in trainX.columns:
    if trainX[col].max() > 100:
        cols_to_norm.append(col)


# In[97]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
fillNa(trainX)
trainX[cols_to_norm] = scaler.fit_transform(trainX[cols_to_norm])


# In[98]:


fillNa(testX)
testX[cols_to_norm] = scaler.fit_transform(testX[cols_to_norm])


# In[99]:


subm = pd.DataFrame()
subm['SK_ID_CURR'] = sk_id
subm.head()
subm.reset_index()
subm.head()


# In[100]:


print(trainX.shape)
y_train = np.array(y_train)
print(y_train.shape)


# In[101]:


finalTrainX = trainX
finalTrainY = y_train


# In[102]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(trainX, y_train, test_size=0.2)


# # XGBoost

# In[103]:


import xgboost as xgb


# In[104]:


import os
os.environ['http_proxy']="http_proxy"
os.environ['https_proxy']="https_proxy"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='3'


# In[105]:


num_class = len(np.unique(y_train))
print(num_class)


# In[106]:


params = {'max_depth': 10, 'n_estimators': 500, 'tree_method': 'gpu_hist', 'n_gpus':1 , 'backend':'h2o4gpu',
              'random_state': 123 , 'n_jobs': -1, 'predictor': 'gpu_predictor', 'verbose':1, 'eval_metric': 'logloss'}


# In[107]:


model = xgb.XGBClassifier(**params)


# In[108]:


model.fit(X_train, y_train, early_stopping_rounds=50, eval_set=[(X_test, y_test)])


# In[157]:


model.fit(finalTrainX, finalTrainY)


# In[158]:


result = model.predict_proba(testX)


# In[159]:


result[:, -1]


# In[160]:


len(sk_id)


# In[161]:


# col = [building_id, damage_grade]
subm = pd.DataFrame()
subm['SK_ID_CURR'] = sk_id
ans = pd.DataFrame(data=result[:, -1], columns=['TARGET'])
subm2 = pd.concat([subm, ans], axis=1)
subm2.to_csv('../Submissions/final_5.csv', index=False)
# subm.head(100)


# In[77]:


subm.shape


# In[78]:


subm2.shape


# In[75]:


result.shape


# In[72]:


ans.shape

