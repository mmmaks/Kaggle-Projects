
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd


# In[12]:


df_train = pd.read_csv('../Data/train.csv')
df_test = pd.read_csv('../Data/test.csv')


# In[3]:


# from matplotlib import pyplot
# from mpl_toolkits.mplot3d import Axes3D

# fig = pyplot.figure()
# ax = Axes3D(fig)

# ax.scatter(temp.pickup_longitude, temp.pickup_latitude, temp.fare_amount)
# pyplot.show()


# In[13]:


df_train.shape


# In[14]:


len(df_train[df_train['fare_amount'] > 50])


# In[15]:


def remove_outliers(df):
    df = df[df['fare_amount'] < 200]
    df = df[df['fare_amount'] > 0]
    
    df = df[df['pickup_longitude'] >= -85]
    df = df[df['pickup_longitude'] <= -70]
    df = df[df['dropoff_longitude'] >= -90]
    df = df[df['dropoff_longitude'] <= -65]
    
    df = df[df['pickup_latitude'] >= 40]
    df = df[df['pickup_latitude'] <= 46]
    df = df[df['dropoff_latitude'] >= 35]
    df = df[df['dropoff_latitude'] <= 50]
    
    df = df[df['passenger_count'] <= 6]
    df = df[df['passenger_count'] > 0]
    return df


# In[16]:


len(df_train[df_train['fare_amount'] > 200])


# In[17]:


temp = remove_outliers(df_train)


# In[18]:


df_train['pickup_longitude'].head()


# In[19]:


temp.head()


# In[20]:


def calc_day_of_week(datetime):
    ret = []
    for x in datetime:
        d = np.int(x.split(' ')[0].split('-')[2])
        m = np.int(x.split(' ')[0].split('-')[1])
        y = np.int(x.split(' ')[0].split('-')[0])
        ret.append((d + np.int(2.6*m - 0.2) - 2*np.int(y/100) + y + np.int(y/4) + np.int(y/400))%7)
    return ret


# In[21]:


def add_year_month_hr_week(df):
    df['week_day'] = calc_day_of_week(df['pickup_datetime'])
    df['year'] = [np.int(x.split(' ')[0].split('-')[0]) for x in df.pickup_datetime]
    df['month'] = [np.int(x.split(' ')[0].split('-')[1]) for x in df.pickup_datetime]
    df['hour'] = [np.int(x.split(' ')[1].split(':')[0]) for x in df.pickup_datetime]
    return df


# In[22]:


temp_y = add_year_month_hr_week(temp)


# In[24]:


temp_y.head()


# In[41]:


# y_train = temp_y['fare_amount']
# X_train = temp_y.drop(['key', 'fare_amount', 'pickup_datetime'], axis=1)


# In[25]:


temp_y.shape


# In[26]:


y_train = temp_y['fare_amount']


# In[27]:


X_train = temp_y


# In[45]:


# X_train = temp_y.drop(['key', 'fare_amount', 'pickup_datetime'], axis=1)
# # X_train = temp_y.drop(['key', 'pickup_datetime'], axis=1)
# X_train.head()


# In[28]:


def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    """
    Return distance along great radius between pickup and dropoff coordinates.
    """
    #Define earth radius (km)
    R_earth = 6371
    #Convert degrees to radians
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(np.radians,
                                                             [pickup_lat, pickup_lon, 
                                                              dropoff_lat, dropoff_lon])
    #Compute distances along lat, lon dimensions
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon
    
    #Compute haversine distance
    a = np.sin(dlat/2.0)**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon/2.0)**2
    
    return 2 * R_earth * np.arcsin(np.sqrt(a))


# In[29]:


# X_train['dist'] = sphere_dist(X_train['pickup_latitude'], X_train['pickup_longitude'], 
#                                    X_train['dropoff_latitude'] , X_train['dropoff_longitude'])


# In[30]:


def manhattan(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    return abs(pickup_lat - dropoff_lat) + abs(pickup_lon - dropoff_lon)


# In[31]:


X_train['manhattan'] = manhattan(X_train['pickup_latitude'], X_train['pickup_longitude'], 
                                   X_train['dropoff_latitude'] , X_train['dropoff_longitude'])


# In[32]:


jfk = [40.6413, -73.7781]
la = [40.7769, -73.8740]
wl = [40.7548, -74.0070]
# ts = [40.7590, -73.9845]


# In[33]:


X_train['jfk'] = manhattan(jfk[0], jfk[1], X_train['pickup_latitude'], X_train['pickup_longitude'])
X_train['la'] = manhattan(la[0], la[1], X_train['pickup_latitude'], X_train['pickup_longitude'])
X_train['wl'] = manhattan(wl[0], wl[1], X_train['pickup_latitude'], X_train['pickup_longitude'])
# X_train['ts'] = manhattan(ts[0], ts[1], X_train['pickup_latitude'], X_train['pickup_longitude'])


# In[71]:


# X_train.to_csv('../Modified_data/1.csv', index=False)


# In[72]:


# X_train['d_jfk'] = manhattan(jfk[0], jfk[1], X_train['dropoff_latitude'], X_train['dropoff_longitude'])
# X_train['d_la'] = manhattan(la[0], la[1], X_train['dropoff_latitude'], X_train['dropoff_longitude'])
# X_train['d_wl'] = manhattan(wl[0], wl[1], X_train['dropoff_latitude'], X_train['dropoff_longitude'])


# In[34]:


X_train.head()


# In[75]:


def one_hot_encode(df, column):
    df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
    df.drop([column], axis=1, inplace=True)
    return df


# In[76]:


def label_encoding(df, column):
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(df[column].unique())
    list(le.classes_)
    df[column] = le.transform(df[column])
    return df


# In[35]:


X_ohe = X_train
# cols = ['year', 'month', 'hour', 'week_day']
# cols = ['year']
# for col in cols:
#     X_ohe = label_encoding(X_ohe, col)
#     X_ohe = one_hot_encode(X_ohe, col)


# In[36]:


y_train.head()


# In[37]:


X_ohe.head()


# In[38]:


X_mod = X_ohe #[X_ohe['manhattan'] > 0]


# In[39]:


y_train = X_mod['fare_amount']
X_mod = X_mod.drop(['key', 'fare_amount', 'pickup_datetime'], axis=1)
# X_train = temp_y.drop(['key', 'pickup_datetime'], axis=1)
X_mod.head()


# In[40]:


from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_mod, y_train, test_size=0.1)


# In[41]:


ytrain.head()


# In[42]:


Xtrain.head()


# In[43]:


ytrain.head()


# In[44]:


ytest.shape


# ## Deep Learning

# In[1]:


from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True #allows dynamic growth
config.gpu_options.visible_device_list = "3" #set GPU number
set_session(tf.Session(config=config))


# In[1]:


import pandas as pd
train = pd.read_csv('../Modified_data/more_50.csv')


# In[2]:


y_train = train['fare_amount']
X_train = train.drop(['fare_amount'], axis=1)


# In[3]:


from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_train, y_train, test_size=0.1)


# In[4]:


from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import rmsprop


# In[6]:


Xtrain.head()


# In[117]:


model = Sequential()

model.add(Dense(Xtrain.shape[1], input_dim=Xtrain.shape[1], activation='sigmoid'))
# model.add(BatchNormalization())


model.add(Dense(512, bias_initializer='ones',activation='sigmoid'))
# # model.add(BatchNormalization())
model.add(Dense(512, bias_initializer='ones',activation='sigmoid'))
# model.add(Dense(256, bias_initializer='ones',activation='sigmoid'))
model.add(Dense(128 , bias_initializer='ones', activation='relu'))
# # model.add(BatchNormalization())

model.add(Dense(64, bias_initializer='ones', activation='sigmoid'))
# # model.add(BatchNormalization())

# model.add(Dense(8, bias_initializer='ones', activation='sigmoid'))

model.add(Dense(4, activation='relu'))

model.add(Dense(1))


# In[118]:


model.summary()


# In[119]:


opt = rmsprop(lr=0.0001, decay=1e-6)


# In[120]:


model.compile(loss='mean_squared_error',
              optimizer=opt)
#               metrics=['accuracy'])


# In[121]:


batch_size = 20024
epochs = 30


# In[122]:


# model.predict(Xtrain)


# In[125]:


# model.fit(Xtrain, ytrain,
#           batch_size=batch_size,
#           epochs=epochs,
#           validation_data=(Xtest, ytest))


# ## SKlearn Linear regression

# In[126]:


from sklearn import linear_model
from sklearn.metrics import mean_squared_error


# In[127]:


regr = linear_model.LinearRegression()


# In[128]:


regr.fit(Xtrain, ytrain)


# In[129]:


ytest_pred = regr.predict(Xtest)


# In[130]:


print("Mean squared error: %.2f"
      % mean_squared_error(ytest_pred, ytest))


# ## XGBoost

# In[45]:


import xgboost as xgb


# In[46]:


import os
os.environ['http_proxy']=http_proxy
os.environ['https_proxy']=https_proxy
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='3'


# In[47]:


params = {'max_depth': 8, 'n_estimators': 2000, 'tree_method': 'gpu_hist', 'n_gpus':1 , 'backend':'h2o4gpu', 'objective':'reg:linear',
              'random_state': 123 , 'n_jobs': -1, 'predictor': 'gpu_predictor', 'verbose':1, 'eval_metric': 'rmse'}


# In[48]:


model = xgb.XGBRegressor(**params)


# In[49]:


model.fit(Xtrain, ytrain, early_stopping_rounds=100, eval_set=[(Xtest, ytest)])


# ## Prediction

# In[50]:


# test_data = remove_outliers(df_test)
test_data = add_year_month_hr_week(df_test)


# In[51]:


df_test.head()


# In[52]:


test_data.head()


# In[53]:


key = test_data['key']


# In[54]:


Xtest = test_data.drop(['key', 'pickup_datetime'], axis=1)


# In[55]:


Xtest.head()


# In[56]:


# Xtest['dist'] = sphere_dist(Xtest['pickup_latitude'], Xtest['pickup_longitude'], 
#                                    Xtest['dropoff_latitude'] , Xtest['dropoff_longitude'])
Xtest['manhattan'] = manhattan(Xtest['pickup_latitude'], Xtest['pickup_longitude'], 
                                   Xtest['dropoff_latitude'] , Xtest['dropoff_longitude'])


# In[57]:


Xtest['jfk'] = manhattan(jfk[0], jfk[1], Xtest['pickup_latitude'], Xtest['pickup_longitude'])
Xtest['la'] = manhattan(la[0], la[1], Xtest['pickup_latitude'], Xtest['pickup_longitude'])
Xtest['wl'] = manhattan(wl[0], wl[1], Xtest['pickup_latitude'], Xtest['pickup_longitude'])
# Xtest['ts'] = manhattan(ts[0], ts[1], Xtest['pickup_latitude'], Xtest['pickup_longitude'])


# In[58]:


# Xtest[Xtest['manhattan'] <= 0]


# In[59]:


Xtest.head()


# In[60]:


Xtrain.head()


# In[61]:


Xtest = Xtest[Xtrain.columns]


# In[22]:


# Xtest['key'] = key


# In[23]:


# Xtest.to_csv('../Modified_data/test1.csv', index=False)


# In[62]:


print(Xtrain.columns)
Xtest.shape


# In[63]:


pred = model.predict(Xtest)


# In[64]:


pred


# In[65]:


submission = pd.DataFrame({
        "key": key,
        "fare_amount": pred
})


# In[66]:


submission.head()


# In[67]:


submission.to_csv('../Submissions/X5.csv', index=False)

