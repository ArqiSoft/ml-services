'''
Keras implementation of deep embedder to improve clustering, inspired by:
"Unsupervised Deep Embedding for Clustering Analysis" (Xie et al, ICML 2016)
Definition can accept somewhat custom neural networks. Defaults are from paper.
'''
import sys
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import SGD
from sklearn.preprocessing import normalize
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE
from .learner.models import lw_autoencoder_model
import tensorflow as tf
import cProfile
import os


if (sys.version[0] == 2):
    import cPickle as pickle
else:
    import pickle

class ClusteringLayer(Layer):
    '''
    Clustering layer which converts latent space Z of input layer
    into a probability vector for each cluster defined by its centre in
    Z-space. Use Kullback-Leibler divergence as loss, with a probability
    target distribution.
    # Arguments
        output_dim: int > 0. Should be same as number of clusters.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        alpha: parameter in Student's t-distribution. Default is 1.0.
    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(nb_samples, output_dim)`.
    '''

    def __init__(self, output_dim, input_dim=None, weights=None, alpha=1.0, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.alpha = alpha
        # kmeans cluster centre locations
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(ClusteringLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = K.variable(self.initial_weights)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        q = 1.0 / (1.0 + K.sqrt(K.sum(K.square(K.expand_dims(x, 1) - self.W), axis=2)) ** 2 / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'input_dim': self.input_dim}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DeepEmbeddingClustering(object):
    def __init__(self,
                 n_clusters,
                 input_dim,
                 encoded=None,
                 decoded=None,
                 alpha=1.0,
                 drop_out=0.5,
                 pretrained_weights=None,
                 cluster_centres=None,
                 batch_size=1024,
                 output_path='',
                 **kwargs):

        super(DeepEmbeddingClustering, self).__init__()

        self.n_clusters = n_clusters
        self.input_dim = input_dim
        self.encoded = encoded
        self.decoded = decoded
        self.alpha = alpha
        self.device='/gpu:0'
        self.drop_out = drop_out
        self.pretrained_weights = pretrained_weights
        self.cluster_centres = cluster_centres
        self.batch_size = batch_size
        self.sub_folder = 'clust_' + str(self.n_clusters)
        
        if output_path:
            if not os.path.isdir(output_path):
                os.mkdir(output_path)
            self.sub_folder = os.path.join(output_path, self.sub_folder)

        if not os.path.isdir(self.sub_folder):
            os.mkdir(self.sub_folder)

        self.learning_rate = 0.1
        self.iters_lr_update = 20000
        self.lr_change_rate = 0.1

        # greedy layer-wise training before end-to-end training:

        self.input_layer = Input(shape=(self.input_dim,), name='input')
        encoders_dims = [input_dim, 500, 500, 2000, 10]
        lw_autoencoder = lw_autoencoder_model(input_dim=self.input_dim,drop_out=self.drop_out,
                                              device=self.device,l_rate=self.learning_rate,
                                              n_clusters=self.n_clusters)
        self.autoencoder = lw_autoencoder['model']
        self.layer_wise_autoencoders = lw_autoencoder['layer_wise_autoencoders']
        self.encoder = lw_autoencoder['encoder']
        self.encoders = lw_autoencoder['encoders']

        if cluster_centres is not None:
            assert cluster_centres.shape[0] == self.n_clusters
            assert cluster_centres.shape[1] == self.encoder.layers[-1].output_dim

        if self.pretrained_weights is not None:
            self.autoencoder.load_weights(self.pretrained_weights)

    def p_mat(self, q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def initialize(self, X, save_autoencoder=False, layerwise_pretrain_iters=50000, finetune_iters=100000):
        if self.pretrained_weights is None:

            iters_per_epoch = int(len(X) / self.batch_size)
            layerwise_epochs = max(int(layerwise_pretrain_iters / iters_per_epoch), 1)
            finetune_epochs = max(int(finetune_iters / iters_per_epoch), 1)

            print('layerwise pretrain')
            current_input = X
            lr_epoch_update = max(1, self.iters_lr_update / float(iters_per_epoch))

            def step_decay(epoch):
                initial_rate = self.learning_rate
                factor = int(epoch / lr_epoch_update)
                lr = initial_rate / (10 ** factor)
                return lr

            lr_schedule = LearningRateScheduler(step_decay)
            reduce_lr = ReduceLROnPlateau(
                monitor='loss', factor=0.9, patience=25, min_lr=0, verbose=1
            )
            earlystopping = EarlyStopping(
                monitor='loss', min_delta=0.0001, patience=100, verbose=1,
                mode='auto'
            )

            with tf.device(('/{}'.format(self.device))):
                for i, autoencoder in enumerate(self.layer_wise_autoencoders):
                    if i > 0:
                        weights = self.encoders[i - 1].get_weights()
                        dense_layer = Dense(self.encoders_dims[i], input_shape=(current_input.shape[1],),
                                            activation='relu', weights=weights,
                                            name='encoder_dense_copy_%d' % i)
                        encoder_model = Sequential([dense_layer])
                        encoder_model.compile(loss='mse', optimizer=SGD(lr=self.learning_rate, decay=0, momentum=0.9))
                        current_input = encoder_model.predict(current_input)
                    autoencoder.fit(current_input, current_input,
                                    batch_size=self.batch_size, epochs=layerwise_epochs,
                                    callbacks=[lr_schedule, reduce_lr, earlystopping])
                    self.autoencoder.layers[i].set_weights(autoencoder.layers[1].get_weights())
                    self.autoencoder.layers[len(self.autoencoder.layers) - i - 1].set_weights(
                        autoencoder.layers[-1].get_weights())

                print('Finetuning autoencoder')

                # update encoder and decoder weights:
                self.autoencoder.fit(X, X, batch_size=self.batch_size, epochs=finetune_epochs,
                                     callbacks=[lr_schedule, reduce_lr, earlystopping])

                if save_autoencoder:
                    self.autoencoder.save_weights('autoencoder.h5')
        else:
            print('Loading pretrained weights for autoencoder.')
            self.autoencoder.load_weights(self.pretrained_weights)

        # update encoder, decoder
        # TODO: is this needed? Might be redundant...
        for i in range(len(self.encoder.layers)):
            self.encoder.layers[i].set_weights(self.autoencoder.layers[i].get_weights())

        # initialize cluster centres using k-means
        print('Initializing cluster centres with k-means.')
        with tf.device(('/{}'.format(self.device))):
            if self.cluster_centres is None:
                kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
                self.y_pred = kmeans.fit_predict(self.encoder.predict(X))
                self.cluster_centres = kmeans.cluster_centers_

            # prepare DEC model
            # self.DEC = Model(inputs=self.input_layer,
            #                 outputs=ClusteringLayer(self.n_clusters,
            #                                        weights=self.cluster_centres,
            #                                        name='clustering')(self.encoder))
            self.DEC = Sequential([self.encoder,
                                   ClusteringLayer(self.n_clusters,
                                                   weights=self.cluster_centres,
                                                   name='clustering')])
            self.DEC.compile(loss='kullback_leibler_divergence', optimizer='adadelta')
            self.DEC_coord = Sequential([self.encoder,
                                   ClusteringLayer(2,
                                                   weights=self.cluster_centres,
                                                   name='clustering')])
            self.DEC_coord.compile(loss='kullback_leibler_divergence', optimizer='adadelta')
        return

    def cluster_acc(self, y_true, y_pred):
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, w

    def cluster(self, X, y=None,
                tol=0.1, update_interval=None,
                iter_max=1e6,
                save_interval=None,
                **kwargs):

        if update_interval is None:
            # 1 epochs
            update_interval = X.shape[0] / self.batch_size
        print('Update interval', update_interval)

        if save_interval is None:
            # 50 epochs
            save_interval = X.shape[0] / self.batch_size * 500
        print('Save interval', save_interval)

        assert save_interval >= update_interval

        train = True
        iteration, index = 0, 0
        self.accuracy = []
        with tf.device(('/{}'.format(self.device))):
            while train:
                sys.stdout.write('\r')
                # cutoff iteration
                if iter_max < iteration:
                    print('Reached maximum iteration limit. Stopping training.')
                    return self.y_pred

                # update (or initialize) probability distributions and propagate weight changes
                # from DEC model to encoder.
                if iteration % update_interval == 0:
                    self.q = self.DEC.predict(X, verbose=0)
                    self.coord = self.DEC_coord.predict(X, verbose = 0)
                    self.p = self.p_mat(self.q)

                    y_pred = self.q.argmax(1)
                    delta_label = ((y_pred == self.y_pred).sum().astype(np.float32) / y_pred.shape[0])
                    if y is not None:
                        acc = self.cluster_acc(y, y_pred)[0]
                        self.accuracy.append(acc)
                        print('Iteration ' + str(iteration) + ', Accuracy ' + str(np.round(acc, 5)))
                    else:
                        print(str(np.round(delta_label * 100, 5)) + '% change in label assignment')

                    if delta_label < tol:
                        print('Reached tolerance threshold. Stopping training.')
                        train = False
                        continue
                    else:
                        self.y_pred = y_pred

                    for i in range(len(self.encoder.layers)):
                        self.encoder.layers[i].set_weights(self.DEC.layers[0].layers[i].get_weights())
                    self.cluster_centres = self.DEC.layers[-1].get_weights()[0]

                # train on batch
                sys.stdout.write('Iteration %d, ' % iteration)
                if (index + 1) * self.batch_size > X.shape[0]:
                    loss = self.DEC.train_on_batch(X[index * self.batch_size::], self.p[index * self.batch_size::])
                    index = 0
                    sys.stdout.write('Loss %f' % loss)
                else:
                    loss = self.DEC.train_on_batch(X[index * self.batch_size:(index + 1) * self.batch_size],
                                                   self.p[index * self.batch_size:(index + 1) * self.batch_size])
                    sys.stdout.write('Loss %f' % loss)
                    index += 1

                # save intermediate
                if iteration % save_interval == 0:
                    z = self.encoder.predict(X)
                    #pca = PCA(n_components=3,whiten=True,svd_solver='arpack',random_state=8).fit(z)
                    #z_2d = pca.transform(z)
                    #clust_2d = pca.transform(self.cluster_centres)
                    tsne=TSNE(n_components=3,n_jobs=2)
                    z_2d = tsne.fit_transform(z)
                    clust_2d = tsne.fit_transform(self.cluster_centres)
                    # save states for visualization
                    pickle.dump({'z_2d': z_2d, 'clust_2d': clust_2d, 'q': self.q, 'p': self.p, 'coord': self.coord},
                                open(str(self.sub_folder)+'/c' + str(iteration) + '.pkl', 'wb'))
                    # save DEC model checkpoints
                    self.DEC.save(str(self.sub_folder)+'/DEC_model_' + str(iteration) + '.h5')

                iteration += 1
                sys.stdout.flush()
        return