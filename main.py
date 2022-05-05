import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import datetime

train = pd.read_csv('../mnist/train.csv')
X, y = np.reshape(np.array(train.iloc[:,1:]), (-1, 28,28 ,1)), train.iloc[:,0]

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, random_state=42, test_size = .2, shuffle=True)

#data augmentation
from scipy.ndimage.interpolation import shift

# Method to shift the image by given dimension
def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])


# Creating Augmented Dataset
X_train_augmented = [image for image in X_train]
y_train_augmented = [image for image in y_train]

for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
     for image, label in zip(X_train, y_train):
             X_train_augmented.append(shift_image(image, dx, dy))
             y_train_augmented.append(label)


# Shuffle the dataset
shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = np.array(X_train_augmented)[shuffle_idx]
y_train_augmented = np.array(y_train_augmented)[shuffle_idx]


# Creating Augmented Dataset
X_val_augmented = [image for image in X_val]
y_val_augmented = [image for image in y_val]


for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
     for image, label in zip(X_val, y_val):
             X_test_augmented.append(shift_image(image, dx, dy))
             y_test_augmented.append(label)

X_train_augmented  = tf.convert_to_tensor(X_train_augmented , dtype = 'float32')
X_val_augmented  = tf.convert_to_tensor(X_val_augmented , dtype = 'float32')
y_train_augmented  = tf.convert_to_tensor(y_train_augmented , dtype = 'float32')
y_val_augmented  = tf.convert_to_tensor(y_val_augmented , dtype = 'float32')


a = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Resizing(height =32,width=32), ])

a(tf.expand_dims(X_train[0], 0))


fig, ax = plt.subplots(10, 10, figsize = (20,20))
for i, xy in enumerate(zip(X_train[:100], y_train[:100])):
    x, y = xy
    ax[i//10, i%10].imshow(x, cmap = plt.cm.gray)
    ax[i//10, i%10].set_title(y.numpy().astype('int32'))
    ax[i//10, i%10].set_xticks([])
    ax[i//10, i%10].set_yticks([])

from Ensnet import *

ens = Ensnet(10)

ens.compile(optimizer = tf.keras.optimizers.Adam(.0001),
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics = tf.keras.metrics.Accuracy()
           )
ens.fit(X_train_augmented, y_train_augmented, validation_data=(X_val_augmented, y_val_augmented), batch_size = 256,
       callbacks = [tf.keras.callbacks.EarlyStopping(patience = 10, restore_best_weights=True, monitor = 'val_mean_loss')],
       epochs = 10000)

pred = ens.predict(test)
