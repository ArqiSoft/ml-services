from keras import Sequential, Model, Input
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, BatchNormalization, Concatenate
from keras.models import load_model
import numpy as np
import pandas as pd
import os
from keras.optimizers import Nadam, RMSprop, Adam
from keras.regularizers import l2
from keras.utils import plot_model
from rdkit import Chem
from processor import sdf_to_csv
from sklearn.externals import joblib
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from learner.plotters import plot_train_history
from keras.layers import PReLU

path_to_sdf = 'C:\PycharmProjects\ml.services\Data\EstrogenCHEMBL206-Binding-K.sdf'
path_to_sdf_test = 'C:\PycharmProjects\ml.services\Data\EstrogenCHEMBL206-Binding-K.sdf'

classname = 'cut_off_activity'


def get_mapper(dataframe):
    beta = 0.0
    opt = Nadam(lr=0.001)
    print(dataframe.head(10))
    x_train,x_test = train_test_split(dataframe, random_state=6,test_size=0.2)
    scaler = MinMaxScaler()
    var_thresh = VarianceThreshold()
    var_thresh = var_thresh.fit(x_train)
    x_train = var_thresh.transform(x_train)
    x_test = var_thresh.transform(x_test)
    scaler = scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    input = Input(x_train.shape[1:])

    batch_norm_1 = BatchNormalization()
    batch_norm_2 = BatchNormalization()
    batch_norm_3 = BatchNormalization()
    batch_norm_4 = BatchNormalization()
    batch_norm_5 = BatchNormalization()
    batch_norm_6 = BatchNormalization()
    batch_norm_7 = BatchNormalization()
    batch_norm_8 = BatchNormalization()
    batch_norm_9 = BatchNormalization()
    batch_norm_10 = BatchNormalization()
    batch_norm_11 = BatchNormalization()
    batch_norm_12 = BatchNormalization()
    batch_norm_neck = BatchNormalization()

    dense_input = Dense(x_train.shape[1:][0], kernel_regularizer=l2(beta))
    dense_1 = Dense(int(x_train.shape[1:][0]/2), kernel_regularizer=l2(beta))
    dense_2 = Dense(int(x_train.shape[1:][0]/4), kernel_regularizer=l2(beta))
    dense_3 = Dense(256, kernel_regularizer=l2(beta))
    dense_4 = Dense(128, kernel_regularizer=l2(beta))
    dense_5 = Dense(64, kernel_regularizer=l2(beta))
    dense_6 = Dense(64, kernel_regularizer=l2(beta))
    dense_7 = Dense(128, kernel_regularizer=l2(beta))
    dense_8 = Dense(256, kernel_regularizer=l2(beta))
    dense_9 = Dense(int(x_train.shape[1:][0]/4), kernel_regularizer=l2(beta))
    dense_10 = Dense(int(x_train.shape[1:][0]/2), kernel_regularizer=l2(beta))
    dense_11 = Dense(x_train.shape[1:][0], kernel_regularizer=l2(beta))
    desc_decoder = Dense(x_train.shape[1:][0], activation="linear", kernel_regularizer=l2(beta))
    neck = Dense(3,kernel_regularizer=l2(beta))
    p_relu= PReLU()
    p_relu2= PReLU()
    p_relu3= PReLU()
    p_relu4= PReLU()
    p_relu5= PReLU()
    p_relu6= PReLU()
    p_relu7= PReLU()
    p_relu8= PReLU()
    p_relu9= PReLU()
    p_relu10= PReLU()
    p_relu11= PReLU()
    p_relu12= PReLU()
    p_relu_neck= PReLU()

    layer1 = batch_norm_1(p_relu(dense_input(input)))
    layer2 = batch_norm_2(p_relu2(dense_1(layer1)))
    layer3 = batch_norm_3(p_relu3(dense_2(layer2)))
    neck_out = p_relu_neck(neck(layer3))
    layer10 = batch_norm_10(p_relu4(dense_9(batch_norm_neck(neck_out))))
    layer11 = batch_norm_11(p_relu5(dense_10(layer10)))
    layer12 = batch_norm_12(p_relu6(dense_11(layer11)))

    decoded_descs = desc_decoder(layer12)

    autoencoder = Model(input,
                        decoded_descs)
    print(autoencoder.summary())
    plot_model(autoencoder,to_file='model_graph.png')


    autoencoder.compile(optimizer=opt, loss='mean_squared_error')
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.7, patience=3, min_lr=0, verbose=1, epsilon=0.00001
    )
    earlystopping = EarlyStopping(
        monitor='val_loss', min_delta=0.00001, patience=15, verbose=1,
        mode='auto'
    )
    # Save the model for best validation loss
    checkpointer = ModelCheckpoint(
        filepath='checkpoint.h5', monitor='val_loss', verbose=1,
        save_best_only=True
    )

    model_history_tmp = autoencoder.fit(x_train,
                                        x_train,
        validation_data=(
            x_test,
            x_test
        ), epochs=10000, batch_size=32,
        callbacks=[checkpointer, earlystopping, reduce_lr],
        shuffle=False, verbose=0
    )

    plot_train_history(
        model_history_tmp, 'compressor_0_1', '')

    # load the best model base on validation results for this fold
    autoencoder = load_model(
        'checkpoint.h5')

    latent_to_map = Model(input, neck_out)
    latent_to_map.save('smi2lat.h5' )

    return latent_to_map,var_thresh,scaler

fptype = [{'Type': 'DESC'},{'Type': 'SEQ'}]
dataframe = sdf_to_csv(path_to_sdf,fptype=fptype,class_name_list=classname)
dataframe = dataframe.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna(
            axis=0, how='any').reset_index(drop=True)

x_features = dataframe.ix[:, :-1]
print(x_features)
lat2coord,var_thresh,scaler = get_mapper(x_features)

data_list = []
for id, row in dataframe.iterrows():
    lis = []
    lis.extend(list(lat2coord.predict(scaler.transform(var_thresh.transform(np.reshape(row[:-1].values,(1,455)))))[0]))
    print((var_thresh.transform(np.reshape(row[:-1].values,(1,455))))[0])
    print(list(lat2coord.predict(scaler.transform(var_thresh.transform(np.reshape(row[:-1].values,(1,455)))))[0]))
    data_list.append(lis)

dataframe_to_concat = pd.DataFrame(data_list,columns=[str(x)+'map' for x in range(3)])
horizontal_stack = pd.concat([dataframe,dataframe_to_concat], axis=1)
print(horizontal_stack)
horizontal_stack.to_csv('latent_descs_with_2d.csv',index=False,header=False,sep=' ')