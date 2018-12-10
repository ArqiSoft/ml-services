import os
from os import listdir
import random
from keras.models import load_model

from general_helper import coeff_determination
from learner.evolution.sascorer import calculateScore

from learner.seq2seq.sampler import latent_to_smiles
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.externals import joblib


def get_models(folder_path):

    models_paths = [os.path.join(folder_path, x) for x in listdir(folder_path) if x.split('.')[-1] == 'h5']
    transformer_paths = [os.path.join(folder_path, x) for x in listdir(folder_path) if x.split('.')[-1] == 'sav']
    models = []
    transformers = []
    for transformer in transformer_paths:
        trans = joblib.load(transformer)
        transformers.append(trans)
    for path_to_model in models_paths:
        model_base = load_model(
            path_to_model,
            custom_objects={'coeff_determination': coeff_determination}
        )
        models.append(model_base)

    return transformers, models_paths

def smiles_to_mol(smiles):

    molecule = Chem.MolFromSmiles(smiles)
    return molecule

def predict_property(molecule,transformers,models):

    pass

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

