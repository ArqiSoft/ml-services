import os

from keras import Model
from keras.layers import LSTM, Bidirectional, BatchNormalization, Input, Concatenate, Dense
from keras.models import load_model, Sequential

def get_cpu_smi2lat_model(gpu_model_path,lstm_dim,neck_dim):

    gpu_smi2lat = load_model(gpu_model_path)
    gpu_smi2lat.summary()
    gpu_smi2lat.save_weights('weights.gpu')
    batch_norm1 = BatchNormalization()
    encoder_inputs = Input(shape=(100, 41))
    resolver = Bidirectional(LSTM(lstm_dim, return_sequences=True, return_state=True,recurrent_activation='sigmoid'))

    encoder_inputs_resolved, resolver_state_h_forward, resolver_state_c_forward, \
    resolver_state_h_backward, resolver_state_c_backward, = resolver((encoder_inputs))

    encoder = Bidirectional(LSTM(lstm_dim, return_state=True,recurrent_activation='sigmoid'))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(batch_norm1(encoder_inputs_resolved))

    resolver_states = Concatenate(axis=-1)([resolver_state_h_forward, resolver_state_c_forward,
                                            resolver_state_h_backward, resolver_state_c_backward])
    encoder_states = Concatenate(axis=-1)([forward_h, forward_c, backward_h, backward_c])

    states = Concatenate(axis=-1)([resolver_states, encoder_states])

    neck = Dense(neck_dim, kernel_initializer='lecun_normal', activation="relu", name='neck')

    neck_outputs_b = neck(states)

    cpu_smi2lat_model = Model(encoder_inputs, neck_outputs_b)

    cpu_smi2lat_model.load_weights('weights.gpu')

    try:
        os.remove('weights.gpu')
    except OSError:
        pass

    cpu_smi2lat_model.summary()

    return cpu_smi2lat_model

get_cpu_smi2lat_model('smi2lat.h5',128,256).save('smi2lat_cpu')

