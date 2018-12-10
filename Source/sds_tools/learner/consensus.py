import numpy as np
from keras.models import Sequential
from keras.optimizers import Nadam
from keras.layers import Dense
from sklearn.model_selection import KFold
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

def keras_cons(x,y,device = '/gpu:0'):
    with tf.device(('/{}'.format(device))):
        model = Sequential()
        model.add(Dense(1,input_shape=(x.shape[1],), activation='linear',
                        name='Output', kernel_initializer='random_uniform',
                        use_bias=False))
        model.compile(loss = "mse", optimizer = Nadam(lr=0.01))

        kfold = KFold(n_splits=3, random_state=42, shuffle=True)
        weights = np.ndarray((1,x.shape[1]))
        for train_idx, valid_idx in kfold.split(x, y):
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=15, min_lr=0.00001,
                verbose=1
            )
            stopping = EarlyStopping(
                monitor='val_loss', min_delta=0.01, patience=35, verbose=1,
                mode='auto')

            model.fit(x[train_idx, :], y[train_idx],
                      validation_data=(x[valid_idx], y[valid_idx]),
                      nb_epoch=10001, batch_size=32, shuffle=False,
                      callbacks=[reduce_lr,stopping])
            new_weights = np.asarray(model.get_weights()).reshape((1,x.shape[1]))
            weights = np.append(weights,new_weights,axis=0)

        mean_weights = np.mean(weights[1:],axis=0)
    return mean_weights
