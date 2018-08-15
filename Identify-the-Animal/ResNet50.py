import os

from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Sequential, Model
from keras.layers import MaxPooling2D, Dense, Activation, GlobalAveragePooling2D
import pandas as pd
import numpy as np

from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True #allows dynamic growth
#config.gpu_options.visible_device_list = "0" #set GPU number
set_session(tf.Session(config=config))

import glob
x_train = []
path = glob.glob('data/train/*')
path = [p.split('/') for p in path]
print(path[0])

df = pd.read_csv('data/train.csv')
print(df.head())
x_train = []
x_valid = []
for img in df['Image_id']:
    p = path[0][0] + '/' + path[0][1] + '/' + img
    temp = int(img.split('-')[1].split('.')[0])
    i = image.load_img(p, target_size = (224, 224))
    a = image.img_to_array(i)
    a = np.expand_dims(a, axis = 0)
    a = preprocess_input(a)
    if temp <= 10000:
        x_train.append(a)
    else:
        x_valid.append(a)
x_train = np.vstack(x_train)
x_valid = np.vstack(x_valid)

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

base_model = ResNet50(weights='imagenet',include_top=False)
X = base_model.output
X = GlobalAveragePooling2D()(X)
pred = Dense(30, activation='softmax')(X)
model = Model(input=base_model.input, output=pred)

for layer in base_model.layers:
    layer.trainable=False

# model = Sequential()
# model.add(GlobalAveragePooling2D(input_shape=precomputed_features.shape[1:]))
# model.add(Dense(30))
# model.add(Activation('softmax'))


import keras
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

df = pd.read_csv('data/train.csv')
list1 = list(set(df['Animal']))
labelDict = {}
ct = 0
for elem in list1:
    labelDict[elem] = ct
    ct += 1
print(labelDict)

df = pd.read_csv('data/train.csv')
print(df.tail())
# y_train = [labelDict[y] for y in df['Animal']]
y_train = []
y_valid = []
ct = 0
for label in df['Animal']:
    if ct < 10000:
        y_train.append(labelDict[label])
    else:
        y_valid.append(labelDict[label])
    ct += 1
print(len(y_train))
print(len(y_valid))

y_train = keras.utils.to_categorical(y_train, 30)
y_valid = keras.utils.to_categorical(y_valid, 30)

batch_size = 32
epochs = 5
classes = 30

model.fit(x_train,
          y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_valid, y_valid)
          )

df = pd.read_csv('data/test.csv')
print(df.head())
x_test = []
for img in df['Image_id']:
    p = path[0][0] + '/' + 'test' + '/' + img
    i = image.load_img(p, target_size = (224, 224))
    a = image.img_to_array(i)
    a = np.expand_dims(a, axis = 0)
    a = preprocess_input(a)
    x_test.append(a)
x_test = np.vstack(x_test)
#     x_test.append(cv2.resize(cv2.imread(p), (32, 32)))
# x_test = np.array(x_test)
# x_test = x_test.astype('float32')
# x_test /= 255


# predictions = model.predict(x_test)
print(predictions[99])
print(type(predictions))


df = pd.read_csv('data/train.csv')
attr = set(df['Animal'])
attr = sorted(attr)
attr = np.array(attr)
attr

ans = []
for i in range(0, 6000):
    tmp2 = []
    for j, atr in enumerate(attr):
        tmp = predictions[i][labelDict[atr]]
        tmp2.append(tmp)
    ans.append(tmp2)
ans = np.array(ans)

subm = pd.DataFrame()
df = pd.read_csv('data/test.csv')
subm['image_id'] = df.Image_id
label_df = pd.DataFrame(data=ans, columns=attr)
subm = pd.concat([subm, label_df], axis=1)
subm.to_csv('submitCool.csv', index=False)


