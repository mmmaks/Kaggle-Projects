
# coding: utf-8

# In[161]:


import numpy as np
import pandas as pd


# In[162]:


pd.set_option('display.max_columns', None)


# In[163]:


df = pd.read_csv('DATA/df_train_test_b_bb.csv')
df_merge = pd.read_csv('DATA/C_PC_PA_IP.csv')
df_label = pd.read_csv('DATA/labels.csv')


# In[165]:


df_merge.head()


# In[166]:


temp_df = df
df = df.merge(df_merge, on='SK_ID_CURR', how='left')


# In[167]:


for col in df.columns:
    if "_y" in col:
        df[col] = (df[col] + df[col[:-2] + "_x"]) / 2
        df.drop([col[:-2] + "_x"], axis=1, inplace=True)


# In[168]:


df.head()


# In[18]:


# df = df.reset_index()


# In[169]:


ignore_features = ['is_train', 'is_test']
relevant_features = [col for col in df.columns if col not in ignore_features]


# In[170]:


trainX = df[df['is_train'] == 1][relevant_features]
testX = df[df['is_test'] == 1][relevant_features]


# In[171]:


for col in trainX.columns:
    trainX[col] = pd.to_numeric(trainX[col], errors='coerce')
for col in testX.columns:
    testX[col] = pd.to_numeric(testX[col], errors='coerce')


# In[175]:


trainX.head()


# In[173]:


testX = testX.reset_index()


# In[176]:


testX.head()
testX.drop(['index'], axis=1, inplace=True)
# trainX.drop(['index'], axis=1, inplace=True)
testX.head()


# In[177]:


len(trainX.columns) == len(testX.columns)


# In[178]:


y_train = list(trainX.merge(df_label, on='SK_ID_CURR', how='left')['TARGET']).copy()
len(y_train)


# In[179]:


sk_id = testX['SK_ID_CURR']
trainX.drop(['SK_ID_CURR'], axis=1, inplace=True)
testX.drop(['SK_ID_CURR'], axis=1, inplace=True)


# In[180]:


def fillNa(df):
    for col in df.columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)


# In[181]:


df['AMT_ANNUITY'].mean()


# In[182]:


cols_to_norm = []
for col in trainX.columns:
    if trainX[col].max() > 100:
        cols_to_norm.append(col)


# In[183]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
fillNa(trainX)
trainX[cols_to_norm] = scaler.fit_transform(trainX[cols_to_norm])


# In[184]:


fillNa(testX)
testX[cols_to_norm] = scaler.fit_transform(testX[cols_to_norm])


# In[185]:


trainX.head()


# In[186]:


print(trainX.shape)
y_train = np.array(y_train)
print(y_train.shape)


# In[187]:


finalTrainX = trainX
finalTrainY = y_train


# In[188]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(trainX, y_train, test_size=0.2)


# ## LightGBM

# In[189]:


import lightgbm as lgb


# In[190]:


import os
os.environ['http_proxy']="http_proxy"
os.environ['https_proxy']="https_proxy"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='3'
os.unsetenv('CUDA_VISIBLE_DEVICES')


# In[191]:


get_ipython().system('env | grep CUDA')


# In[192]:


num_class = len(np.unique(y_train))
print(num_class)


# In[193]:


lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)


# In[207]:


params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'num_class': 1,
    'metric': {'binary_logloss', 'auc'},
    'num_iterations': 5000,
    'max_bin': 255,
    'max_depth': 11,
    'num_leaves': 90,
    'learning_rate': 0.01,
    'min_data_in_leaf': 60,
    'sparse_threshold': 1.0,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1,
    'gpu_use_dp': 'true',
    'tree_learner': 'feature',
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 3,
    'num_thread': 32,
}


# In[208]:


model = lgb.LGBMClassifier(**params)


# In[209]:


model.fit(X_train, y_train, early_stopping_rounds=200, eval_set=[(X_test, y_test)])


# In[197]:


model.fit(finalTrainX, finalTrainY)


# In[198]:


result = model.predict_proba(testX)


# In[199]:


result[:, -1]


# In[200]:


# col = [building_id, damage_grade]
subm = pd.DataFrame()
subm['SK_ID_CURR'] = sk_id
ans = pd.DataFrame(data=result[:, -1], columns=['TARGET'])
subm2 = pd.concat([subm, ans], axis=1)
subm2.to_csv('Submissions/final_6.csv', index=False)
# subm.head(100)

