import os
import pickle

import numpy as np
import pandas as pd
from keras.models import load_model
from rdkit import Chem, RDLogger
from .preprocessor import Preprocessor

def latent_to_smiles(charset,smiles_len,char_to_int, int_to_char, lat_to_states_model, sample_model, latent,type='2_layers'):
    #decode states and set Reset the LSTM cells with them
    states = lat_to_states_model.predict(latent)
    if type is '2_layers':
        sample_model.layers[1].reset_states(states=[states[0], states[1]])
        sample_model.layers[3].reset_states(states=[states[2], states[3]])
    if type is 'simple':
        sample_model.layers[1].reset_states(states=[states[0], states[1]])
    #Prepare the input char
    startidx = char_to_int["!"]
    samplevec = np.zeros((1,1,len(charset)))
    samplevec[0,0,startidx] = 1
    smiles = ""
    #Loop and predict next char
    for i in range(smiles_len + 1):
        o = sample_model.predict(samplevec)
        sampleidx = np.argmax(o)
        samplechar = int_to_char[sampleidx]
        if samplechar != "E":
            smiles = smiles + int_to_char[sampleidx]
            samplevec = np.zeros((1,1,len(charset)))
            samplevec[0,0,sampleidx] = 1
        else:
            break
    return Preprocessor.unfold_double_characters(smiles)

def slerp_interpolate(latent_1,latent_2,charset,smiles_len,char_to_int,int_to_char,latent_to_states_model,
                      sample_model,type,check=True,dim=128):
    mols_temp = []
    mols1 = []
    latent_1 = np.reshape(latent_1,(dim))
    latent_2 = np.reshape(latent_2,(dim))

    def slerp(p0, p1, t):
        omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
        so = np.sin(omega)
        return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1

    ps = np.array([slerp(latent_1, latent_2, t) for t in np.arange(0.0, 1.0, 0.001)])

    for p in ps:
        p = np.reshape(p,(1,dim))
        smiles = latent_to_smiles(charset,smiles_len,char_to_int,int_to_char,
                                  latent_to_states_model,sample_model,p,type=type)
        if check:
            if smiles not in mols_temp:
                mols_temp.append(smiles)
                if Chem.MolFromSmiles(smiles):
                    mols1.append(smiles)
        else:
            mols1.append(smiles)

    return mols1

def linear_interpolate(latent_1,latent_2,charset,smiles_len,char_to_int,int_to_char,latent_to_states_model,sample_model,type,check=True):
    mols_temp = []
    mols1 = []
    ratios = np.linspace(0, 1, 1000)
    for r in ratios:
        rlatent = (1.0 - r) * latent_1 + r * latent_2
        smiles = latent_to_smiles(charset, smiles_len, char_to_int, int_to_char,
                                  latent_to_states_model, sample_model, rlatent,
                                  type=type)
        if check:
            if smiles not in mols_temp:
                mols_temp.append(smiles)
                if Chem.MolFromSmiles(smiles):
                    mols1.append(smiles)
        else:
            mols1.append(smiles)

    return mols1

def sample_around(stdev,charset,smiles_len,char_to_int, int_to_char, lat_to_states_model, sample_model,latent,type,check=True):
    stdev = stdev
    mols = []
    for i in range(100):
        latent_r = latent * (np.random.normal(1,stdev,latent.shape[1]))
        smiles = latent_to_smiles(charset,smiles_len,char_to_int, int_to_char, lat_to_states_model,
                                  sample_model,latent_r,type=type)
        if check:
            if Chem.MolFromSmiles(smiles) and smiles not in mols:
                mols.append(smiles)
            else:
                pass
        else:
            mols.append(smiles)

    return mols
