from keras import Sequential, Model, Input
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, BatchNormalization, Concatenate
from keras.models import load_model
import numpy as np
from keras.optimizers import Nadam
from keras.regularizers import l2
from keras.utils import plot_model
from rdkit import Chem
import pandas as pd
from processor import sdf_to_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from learner.plotters import plot_train_history

path_to_sdf = 'C:\PycharmProjects\ml.services\Data\EstrogenCHEMBL206-Binding-K.sdf'

classname = 'Tox'
beta = 0.0
opt = Nadam(lr=0.01)
fptype = [{'Type': 'DESC'},
          {'Type': 'MACCS'},
          {'Type': 'ECFC','Size': 2048,'Radius':3},
          {'Type': 'AVALON','Size': 2048}]


dataframe = pd.read_csv('features_0',delimiter=',',header=None)
dataframe = dataframe.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna(
            axis=0, how='any').reset_index(drop=True)

x_train,x_test = train_test_split(dataframe, random_state=6)
x_train = x_train.as_matrix()
x_test = x_test.as_matrix()
scaler = MinMaxScaler()


scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


descs_train = x_train[:, 0:199]
descs_test = x_test[:, 0:199]
maccs_train = x_train[:, 199:366]
maccs_test = x_test[:, 199:366]
ecfc_train = x_train[:, 366:1390]
ecfc_test = x_test[:, 366:1390]
avalon_train = x_train[:, 1390:2414]
avalon_test = x_test[:, 1390:2414]

input_descs = Input(descs_train.shape[1:])
input_maccs = Input(maccs_train.shape[1:])
input_ecfc = Input(ecfc_train.shape[1:])
input_avalon = Input(avalon_train.shape[1:])

batch_norm_1 = BatchNormalization()
batch_norm_2 = BatchNormalization()
batch_norm_3 = BatchNormalization()
batch_norm_4 = BatchNormalization()
batch_norm_5 = BatchNormalization()
batch_norm_6 = BatchNormalization()
batch_norm_7 = BatchNormalization()
batch_norm_8 = BatchNormalization()
batch_norm_9 = BatchNormalization()

desc_compressor = Dense(96, activation="selu",
            kernel_initializer='lecun_normal',kernel_regularizer=l2(beta))
maccs_compressor = Dense(32, activation="selu",
            kernel_initializer='lecun_normal',kernel_regularizer=l2(beta))
ecfc_compressor = Dense(512, activation="selu",
            kernel_initializer='lecun_normal',kernel_regularizer=l2(beta))
avalon_compressor = Dense(512, activation="selu",
            kernel_initializer='lecun_normal',kernel_regularizer=l2(beta))
neck = Dense(32, activation="selu",
            kernel_initializer='lecun_normal',kernel_regularizer=l2(beta))

desc_compressed = batch_norm_1(desc_compressor(input_descs))
maccs_compressed = batch_norm_2(maccs_compressor(input_maccs))
ecfc_compressed = batch_norm_3(ecfc_compressor(input_ecfc))
avalon_compressed = batch_norm_4(avalon_compressor(input_avalon))

concatenated = Concatenate(axis=-1)([desc_compressed,maccs_compressed,ecfc_compressed,avalon_compressed])

neck_outputs = batch_norm_5(neck(concatenated))

desc_decompressor = Dense(96, activation="selu",
            kernel_initializer='lecun_normal',kernel_regularizer=l2(beta))
maccs_decompressor = Dense(32, activation="selu",
            kernel_initializer='lecun_normal',kernel_regularizer=l2(beta))
ecfc_decompressor = Dense(512, activation="selu",
            kernel_initializer='lecun_normal',kernel_regularizer=l2(beta))
avalon_decompressor = Dense(512, activation="selu",
            kernel_initializer='lecun_normal',kernel_regularizer=l2(beta))

desc_decompressed = batch_norm_6(desc_decompressor(neck_outputs))
maccs_decompressed = batch_norm_7(maccs_decompressor(neck_outputs))
ecfc_decompressed = batch_norm_8(ecfc_decompressor(neck_outputs))
avalon_decompressed = batch_norm_9(avalon_decompressor(neck_outputs))

desc_decoder = Dense(199, activation="linear",
            kernel_initializer='lecun_normal',kernel_regularizer=l2(beta))
maccs_decoder = Dense(167, activation="linear",
            kernel_initializer='lecun_normal',kernel_regularizer=l2(beta))
ecfc_decoder = Dense(1024, activation="linear",
            kernel_initializer='lecun_normal',kernel_regularizer=l2(beta))
avalon_decoder = Dense(1024, activation="linear",
            kernel_initializer='lecun_normal',kernel_regularizer=l2(beta))

desc_decoded = desc_decoder(desc_decompressed)
maccs_decoded = maccs_decoder(maccs_decompressed)
ecfc_decoded = ecfc_decoder(ecfc_decompressed)
avalon_decoded = avalon_decoder(avalon_decompressed)

autoencoder = Model([input_descs,input_maccs,input_ecfc,input_avalon],
                    [desc_decoded,maccs_decoded,ecfc_decoded,avalon_decoded])
print(autoencoder.summary())
plot_model(autoencoder,to_file='C:\PycharmProjects\ml.services\Source\\callers and models\comp_tests\model_graph.png')


autoencoder.compile(optimizer=Nadam(lr=0.01), loss='mean_squared_error',loss_weights=[1., 1., 1., 1.])
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=20, min_lr=0, verbose=1, epsilon=0.0001
)
earlystopping = EarlyStopping(
    monitor='val_loss', min_delta=0.0001, patience=50, verbose=1,
    mode='auto'
)
# Save the model for best validation loss
checkpointer = ModelCheckpoint(
    filepath='C:\PycharmProjects\ml.services\Source\\callers and models\comp_tests\checkpoint.h5', monitor='val_loss', verbose=1,
    save_best_only=True
)

model_history_tmp = autoencoder.fit([descs_train,maccs_train,ecfc_train,avalon_train],
                                    [descs_train,maccs_train,ecfc_train,avalon_train],
    validation_data=(
        [descs_test, maccs_test, ecfc_test, avalon_test],
        [descs_test, maccs_test, ecfc_test, avalon_test]
    ), epochs=10000, batch_size=700,
    callbacks=[checkpointer, earlystopping, reduce_lr],
    shuffle=True, verbose=0
)

plot_train_history(
    model_history_tmp, 'compressor_0_1', 'C:\PycharmProjects\ml.services\Source\\callers and models\comp_tests')

# load the best model base on validation results for this fold
autoencoder = load_model(
    'C:\PycharmProjects\ml.services\Source\\callers and models\comp_tests\checkpoint.h5')

before = x_test[2]
np.savetxt('C:\PycharmProjects\ml.services\Source\\callers and models\comp_tests\\before.csv',before,delimiter=' ')
after = autoencoder.predict(x_test[2].reshape(1,2414))
np.savetxt('C:\PycharmProjects\ml.services\Source\\callers and models\comp_tests\\after.csv',after,delimiter=' ')
