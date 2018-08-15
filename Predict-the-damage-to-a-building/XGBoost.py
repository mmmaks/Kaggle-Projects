
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


print(np.__version__)


# In[3]:


df_train = pd.DataFrame()
df_owner = pd.DataFrame()
df_struct = pd.DataFrame()
df_test = pd.DataFrame()


# In[4]:


def initialize_df():
    global df_train 
    df_train = pd.read_csv('../Data/Dataset/train.csv')
    global df_owner 
    df_owner = pd.read_csv('../Data/Dataset/Building_Ownership_Use.csv')
    global df_struct 
    df_struct = pd.read_csv('../Data/Dataset/Building_Structure.csv')
    global df_test 
    df_test = pd.read_csv('../Data/Dataset/test.csv')
initialize_df()


# In[5]:


df_train.columns


# In[6]:


df_owner.columns


# In[7]:


def spl(Ytrain):
    tempList = []
    for y in Ytrain:
        tempList.append(np.int64(y.split(' ')[1]) - 1)
    return tempList

def to_float(Xtrain):
    tempList = []
    for y in Xtrain:
        tempList.append(np.float64(y))
    return tempList


# In[8]:


def one_hot_encode(df, column):
    df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
    df.drop([column], axis=1, inplace=True)
    return df


# In[9]:


def label_encoding(df, column):
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(df[column].unique())
    list(le.classes_)
    df[column] = le.transform(df[column])
    return df


# In[10]:


def prepare(df):
    temp_df = pd.merge(df_struct, df_owner, how = 'left', on = ['building_id', 'district_id', 'vdcmun_id', 'ward_id'])
    df = pd.merge(df, temp_df, how = 'left', on = ['building_id', 'district_id', 'vdcmun_id' ])

    # One-Hot-Encoding
    for col in df.columns:
        if "count" not in col:
            if len(df[col].unique()) < 20 and len(df[col].unique()) > 2:
                print(col)
                if df[col].isnull().any():
                    print(col, df[col].unique())
                    df[col].fillna(0, inplace=True)
                df = label_encoding(df, col)
                df = one_hot_encode(df, col)
    df['floor_diff'] = df['count_floors_pre_eq'] - df['count_floors_post_eq']
    df['height_diff'] = df['height_ft_pre_eq'] - df['height_ft_post_eq']
    df.drop(['count_floors_pre_eq', 'count_floors_post_eq', 'height_ft_pre_eq', 'height_ft_post_eq'], axis=1, inplace=True)
    print(df.columns)
    return df


# In[11]:


initialize_df()
Ytrain = df_train['damage_grade']
df_train.drop(['damage_grade'], axis=1, inplace=True)
df_train_modified = prepare(df_train)
df_train_modified.drop(['building_id', 'district_id', 'ward_id'], axis=1, inplace=True)
# df_train_modified['damage_grade'] = spl(Ytrain)
# df_train_modified


# In[13]:


# tempList = []
# for col in df_train_modified.columns:
# #     print(col)
#     corr = df_train_modified['damage_grade'].corr(df_train_modified[col])
#     tempList.append([col, corr])
# def getKey(item):
#     return -1*item[1]
# sorted(tempList, key=getKey)


# In[12]:


Ytrain = spl(Ytrain)


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train_modified, Ytrain, test_size=0.2)


# In[14]:


print(df_train_modified.shape)
print(X_train.shape)
print(X_test.shape)


# In[15]:


initialize_df()
df_test_modified = prepare(df_test)
b_id = df_test_modified['building_id']
df_test_modified.drop(['building_id', 'district_id', 'ward_id'], axis=1, inplace=True)


# In[16]:


for col in df_test_modified.columns:
    tempList = []
    if df_test_modified[col].dtypes != np.int64:
        if df_test_modified[col].isnull().any():
            print(col, df_test_modified[col].unique())
            df_test_modified[col].fillna(df_test_modified[col].mean(), inplace=True)
        print(col, df_test_modified[col].dtypes)
        df_test_modified[col].astype(int)
#         for val in df_test_modified[col]:
#             tempList.append(np.int64(val))
#         df_test_modified[col] = tempList


# ## XGBoost

# In[17]:


import xgboost as xgb


# In[18]:


import os
os.environ['http_proxy']="http_proxy"
os.environ['https_proxy']="https_proxy"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='3'


# In[19]:


num_class = len(np.unique(y_train))
print(num_class)


# In[20]:


params = {'eval_metric': 'merror', 'max_depth': 8, 'n_estimators': 300, 'tree_method': 'gpu_hist', 'n_gpus':1 , 'backend':'h2o4gpu',
              'random_state': 123 , 'n_jobs': -1, 'predictor': 'gpu_predictor', 'verbose':1, 'num_class': num_class,
         'min_child_weight': 6}
print("here")


# In[21]:


print("Gotcha")


# In[39]:


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


# In[40]:


print(set(df_train_modified.columns) - set(df_test_modified.columns))


# In[41]:


# df_test_modified.columns


# In[42]:


# model = xgb.XGBClassifier(**params)
model = xgb.train(
    params,
    dtrain,
    evals=[(dtest, "Test")],
    num_boost_round=1000,
    early_stopping_rounds=50
)


# In[35]:


finalTestData = xgb.DMatrix(df_test_modified)


# In[36]:


result = model.predict(finalTestData)


# In[46]:


import time
st = time.time()
time.sleep(1)
print(time.time() - st)


# In[47]:


import time
st = time.time()
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=999,
    seed=42,
    nfold=5,
    metrics={'merror'},
    early_stopping_rounds=10
)
print(time.time() - st)


# In[51]:


for eta in [.3, .2, .1, .05, .01, .005]:
    print("ETA : {}".format(eta))
    params['eta'] = eta
    st = time.time()
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=999,
        seed=42,
        nfold=5,
        metrics={'merror'},
        early_stopping_rounds=10
    )
    print("Error for this eta : {}".format(cv_results['test-merror-mean'].min()))
    print("Time taken : {}".format(time.time() - st))


# In[48]:


cv_results
cv_results['test-merror-mean'].min()


# In[29]:


print("Best MAE: {:.2f} with {} rounds".format(
                 model.best_score,
                 model.best_iteration+1))


# In[ ]:


dtest = xgb.DMatrix(df_test_modified)


# In[ ]:


result = model.predict(dtest)


# In[37]:


tempL = []
for res in result:
    tempL.append("Grade " + str(np.int64(res) + 1))
result = tempL


# In[38]:


# col = [building_id, damage_grade]
subm = pd.DataFrame()
subm['building_id'] = b_id
ans = pd.DataFrame(data=result, columns=['damage_grade'])
subm = pd.concat([subm, ans], axis=1)
subm.to_csv('../Submissions/final_subm_9.csv', index=False)
# subm.head(100)

