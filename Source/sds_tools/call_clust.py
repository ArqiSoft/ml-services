from sds_tools.clustering import DeepEmbeddingClustering
#from keras.datasets import mnist
#import logging
from sds_tools.processor import sdf_to_csv
import numpy as np

#logging.basicConfig(level=logging.INFO)

#filepath = '../../test_in/OpenPHACTS/OPS_DRUGBANK_PROPERTIES.sdf'
#filepath = '../test_in/Eu_const/Eu_dataset.sdf'
filepath = '../test_in/Chebi/ChEBI_complete_3star.sdf'
#valuename = 'MolRefract'
valuename = 'Charge'
fptype = [{'type':'FCFC','radius':3,'nbits':512}]
#fptype = [{'type':'DESC'}]

output_path = '../test_out/clusters'
batch_size = 1024

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["OSDR_LOG_FOLDER"]="../logger"

dataframe = sdf_to_csv(filepath,value_name_list=valuename,fptype=fptype)

'''
def get_mnist():
    np.random.seed(1234)  # set seed for deterministic ordering
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_all = np.concatenate((x_train, x_test), axis=0)
    Y = np.concatenate((y_train, y_test), axis=0)
    X = x_all.reshape(-1, x_all.shape[1] * x_all.shape[2])

    p = np.random.permutation(X.shape[0])
    X = X[p].astype(np.float32) * 0.02
    Y = Y[p]
    return X, Y
X, Y = get_mnist()
'''

bins = len(dataframe.columns) - 1
X = dataframe.ix[:, :bins].as_matrix()
#mask = np.isfinite(X).all(axis=1)
#X = X[mask, :]
Y = dataframe[valuename]

c = DeepEmbeddingClustering(n_clusters=4, input_dim=bins,output_path=output_path,batch_size=batch_size)
c.initialize(X, finetune_iters=100000, layerwise_pretrain_iters=50000)
c.cluster(X)