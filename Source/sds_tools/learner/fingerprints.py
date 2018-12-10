"""
Module which contain all data needed to calculate fingerprints/descriptors
parameters. Contain internal codes, names, molstring and header getters.
"""
import os
import pickle

import keras
import numpy as np
import tensorflow
from keras.models import load_model
from rdkit import Chem
from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP
from rdkit.Chem import (
    AllChem, MACCSkeys, PatternFingerprint, RDKFingerprint, LayeredFingerprint,
    rdReducedGraphs, Descriptors, MolToSmiles, DataStructs
)
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.DataStructs.cDataStructs import ConvertToExplicit

from learner.seq2seq.preprocessor import Preprocessor


# all using in OSDR fingerprints
FCFP = 'Feature-Connectivity FingerPrint'
ECFP = 'Extended-Connectivity Fingerprint'
FCFC = 'Feature Connectivity Fingerprint Count vector'
FCFC_CHIRALITY = 'Feature Connectivity Fingerprint Count vector with Chirality'
ECFC = 'Extented Connectivity Fingerprint Count vector'
ECFC_CHIRALITY = 'Extented Connectivity Fingerprint Count vector with Chirality'
ATOM_PAIRS = 'RDKit Hashed Atom Pairs count vector'
MACCS = '166 public MACCS keys + 1 zero bit'
PATTERN = 'Experimental SMARTS patterns-based fingerprint'
AVALON = 'Avalon count FPs from Avalon cheminformatcis toolkit'
RDK = 'RDKit in-house subgraph-based count vector'
LAYERED = 'Layered path-based-FP'
ERG = 'Extended reduced graph approach uses pharmacophore-type'
PHARMA = 'PHARMA'
DESC = 'ALL of the RDKit supported descriptors'
CAN2CAN = 'CAN2CAN'
ENUM2CAN = 'ENUM2CAN'
CAN2ENUM = 'CAN2ENUM'
ENUM2ENUM = 'ENUM2ENUM'

PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))
CAN2CAN_PATH = os.path.join(PROJECT_PATH, 'seq2seq', 'CAN2CAN')
ENUM2CAN_PATH = os.path.join(PROJECT_PATH, 'seq2seq', 'ENUM2CAN')
CAN2ENUM_PATH = os.path.join(PROJECT_PATH, 'seq2seq', 'CAN2ENUM')
ENUM2ENUM_PATH = os.path.join(PROJECT_PATH, 'seq2seq', 'ENUM2ENUM')

CHAR_TO_INT_ENUM2CAN = pickle.load(
    open(os.path.join(ENUM2CAN_PATH, 'char_to_int.obj'), 'rb'))
INT_TO_CHAR_ENUM2CAN = pickle.load(
    open(os.path.join(ENUM2CAN_PATH, 'int_to_char.obj'), 'rb'))
SMILES_LEN_ENUM2CAN = pickle.load(
    open(os.path.join(ENUM2CAN_PATH, 'smiles_len.obj'), 'rb'))
CHARSET_ENUM2CAN = pickle.load(open(os.path.join(ENUM2CAN_PATH, 'charset.obj'), 'rb'))

CHAR_TO_INT_CAN2CAN = pickle.load(
    open(os.path.join(CAN2CAN_PATH, 'char_to_int.obj'), 'rb'))
INT_TO_CHAR_CAN2CAN = pickle.load(
    open(os.path.join(CAN2CAN_PATH, 'int_to_char.obj'), 'rb'))
SMILES_LEN_CAN2CAN = pickle.load(
    open(os.path.join(CAN2CAN_PATH, 'smiles_len.obj'), 'rb'))
CHARSET_CAN2CAN = pickle.load(open(os.path.join(CAN2CAN_PATH, 'charset.obj'), 'rb'))

CHAR_TO_INT_CAN2ENUM = pickle.load(
    open(os.path.join(CAN2ENUM_PATH, 'char_to_int.obj'), 'rb'))
INT_TO_CHAR_CAN2ENUM = pickle.load(
    open(os.path.join(CAN2ENUM_PATH, 'int_to_char.obj'), 'rb'))
SMILES_LEN_CAN2ENUM = pickle.load(
    open(os.path.join(CAN2ENUM_PATH, 'smiles_len.obj'), 'rb'))
CHARSET_CAN2ENUM = pickle.load(open(os.path.join(CAN2ENUM_PATH, 'charset.obj'), 'rb'))

CHAR_TO_INT_ENUM2ENUM = pickle.load(
    open(os.path.join(ENUM2ENUM_PATH, 'char_to_int.obj'), 'rb'))
INT_TO_CHAR_ENUM2ENUM = pickle.load(
    open(os.path.join(ENUM2ENUM_PATH, 'int_to_char.obj'), 'rb'))
SMILES_LEN_ENUM2ENUM = pickle.load(
    open(os.path.join(ENUM2ENUM_PATH, 'smiles_len.obj'), 'rb'))
CHARSET_ENUM2ENUM = pickle.load(open(os.path.join(ENUM2ENUM_PATH, 'charset.obj'), 'rb'))

keras.backend.clear_session()
CAN2CAN_SMI_TO_LAT_MODEL = load_model(os.path.join(CAN2CAN_PATH, 'smi2lat_cpu'))
ENUM2CAN_SMI_TO_LAT_MODEL = load_model(os.path.join(ENUM2CAN_PATH, 'smi2lat_cpu'))
CAN2ENUM_SMI_TO_LAT_MODEL = load_model(os.path.join(CAN2ENUM_PATH, 'smi2lat_cpu'))
ENUM2ENUM_SMI_TO_LAT_MODEL = load_model(os.path.join(ENUM2ENUM_PATH, 'smi2lat_cpu'))
SMI_TO_LAT_GRAPH = tensorflow.get_default_graph()
SMI_TO_LAT_SESSION = keras.backend.get_session()


def fcfp_molstring(molecule, fptype):
    """
    Method for make molstring for fcfp fingerprint

    :param molecule: molecule object
    :param fptype: type, radius and size of fingerprint
    :type fptype: dict
    :return: molstring for fcfp fingerprint
    """
    arr = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(
        AllChem.GetMorganFingerprintAsBitVect(
            molecule, fptype['Radius'], fptype['Size'], useFeatures=True
        ), arr
    )

    return arr


def ecfp_molstring(molecule, fptype):
    """
    Method for make molstring for ecfp fingerprint

    :param molecule: molecule object
    :param fptype: type, radius and size of fingerprint
    :type fptype: dict
    :return: molstring for ecfp fingerprint
    """
    arr = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(
        AllChem.GetMorganFingerprintAsBitVect(
            molecule, fptype['Radius'], fptype['Size'], useFeatures=False
        ), arr
    )

    return arr


def fcfc_molstring(molecule, fptype):
    """
    Method for make molstring for fcfc without chirality fingerprint

    :param molecule: molecule object
    :param fptype: type, radius and size of fingerprint
    :type fptype: dict
    :return: molstring for fcfc without chirality fingerprint
    """
    arr = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(
        AllChem.GetHashedMorganFingerprint(
            molecule, fptype['Radius'], fptype['Size'], useFeatures=True
        ), arr
    )

    return arr


def fcfc_chirality_molstring(molecule, fptype):
    """
    Method for make molstring for fcfc with chirality fingerprint

    :param molecule: molecule object
    :param fptype: type, radius and size of fingerprint
    :type fptype: dict
    :return: molstring for fcfc with chirality fingerprint
    """
    arr = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(
        AllChem.GetHashedMorganFingerprint(
            molecule, fptype['Radius'], fptype['Size'], useFeatures=True,
            useChirality=True
        ), arr
    )

    return arr


def ecfc_molstring(molecule, fptype):
    """
    Method for make molstring for ecfc without chirality fingerprint

    :param molecule: molecule object
    :param fptype: type, radius and size of fingerprint
    :type fptype: dict
    :return: molstring for ecfc without chirality fingerprint
    """
    arr = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(
        AllChem.GetHashedMorganFingerprint(
            molecule, fptype['Radius'], fptype['Size'], useFeatures=False
        ), arr
    )

    return arr


def ecfc_chirality_molstring(molecule, fptype):
    """
    Method for make molstring for ecfc with chirality fingerprint

    :param molecule: molecule object
    :param fptype: type, radius and size of fingerprint
    :type fptype: dict
    :return: molstring for ecfc with chirality fingerprint
    """

    arr = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(
        AllChem.GetHashedMorganFingerprint(
            molecule, fptype['Radius'], fptype['Size'], useFeatures=False,
            useChirality=True
        ), arr
    )

    return arr


def atom_pairs_molstring(molecule, fptype):
    """
    Method for make molstring for atom pairs fingerprint

    :param molecule: molecule object
    :param fptype: type, radius and size of fingerprint
    :type fptype: dict
    :return: molstring for atom pairs fingerprint
    """
    arr = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(
        AllChem.GetHashedAtomPairFingerprint(
            molecule, nBits=fptype['Size'], includeChirality=True
        ), arr
    )

    return arr


def maccs_molstring(molecule, fptype):
    """
    Method for make molstring for maccs fingerprint

    :param molecule: molecule object
    :param fptype: type, radius and size of fingerprint
    :type fptype: dict
    :return: molstring for maccs fingerprint
    """
    arr = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(
        MACCSkeys.GenMACCSKeys(molecule), arr)

    return arr


def pattern_molstring(molecule, fptype):
    """
    Method for make molstring for pattern fingerprint

    :param molecule: molecule object
    :param fptype: type, radius and size of fingerprint
    :type fptype: dict
    :return: molstring for pattern fingerprint
    """
    arr = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(
        PatternFingerprint(molecule, fptype['Size']), arr)

    return arr


def avalon_molstring(molecule, fptype):
    """
    Method for make molstring for avalon fingerprint

    :param molecule: molecule object
    :param fptype: type, radius and size of fingerprint
    :type fptype: dict
    :return: molstring for avalon fingerprint
    """
    arr = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(
        GetAvalonCountFP(molecule, nBits=fptype['Size']), arr)

    return arr


def rdk_molstring(molecule, fptype):
    """
    Method for make molstring for rdk fingerprint

    :param molecule: molecule object
    :param fptype: type, radius and size of fingerprint
    :type fptype: dict
    :return: molstring for rdk fingerprint
    """
    arr = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(
        RDKFingerprint(molecule, fpSize=fptype['Size']), arr)

    return arr


def layered_molstring(molecule, fptype):
    """
    Method for make molstring for layered fingerprint

    :param molecule: molecule object
    :param fptype: type, radius and size of fingerprint
    :type fptype: dict
    :return: molstring for layered fingerprint
    """

    arr = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(
        LayeredFingerprint(molecule, fpSize=fptype['Size']), arr)

    return arr


def erg_molstring(molecule, fptype):
    """
    Method for make molstring for erg fingerprint

    :param molecule: molecule object
    :param fptype: type, radius and size of fingerprint
    :type fptype: dict
    :return: molstring for erg fingerprint
    """

    return rdReducedGraphs.GetErGFingerprint(molecule)


def pharma_molstring(molecule, fptype):
    """
    Method for make molstring for pharma fingerprint

    :param molecule: molecule object
    :param fptype: type, radius and size of fingerprint
    :type fptype: dict
    :return: molstring for pharma fingerprint
    """
    arr = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(
        ConvertToExplicit(
            Generate.Gen2DFingerprint(molecule, Gobbi_Pharm2D.factory)
        ), arr
    )

    return arr


def can2can(molecule, fptype):
    """
    Function returns a latent vector - based feature vector for a molecule
    which is extracted from canonical-to-canonical seq2seq SMILES autoencoder
    :param molecule: rdkit molecule object
    :param fptype: useless here
    :return: latent vector - based feature vector
    """

    smile = Chem.MolToSmiles(molecule, canonical=True)
    smile = Preprocessor.prepare_smiles(smile, SMILES_LEN_CAN2CAN)
    if smile is None:
        return [None for x in range(256)]
    smile_vec = Preprocessor.vectorize(smile, CHARSET_CAN2CAN, CHAR_TO_INT_CAN2CAN, SMILES_LEN_CAN2CAN)[0]

    keras.backend.set_session(SMI_TO_LAT_SESSION)

    with SMI_TO_LAT_GRAPH.as_default():
        latent_vec = CAN2CAN_SMI_TO_LAT_MODEL.predict(smile_vec)[0]

    return latent_vec


def enum2enum(molecule, fptype):
    """
    Function returns a latent vector - based feature vector for a molecule
    which is extracted from enumerated-to-enumerated seq2seq SMILES autoencoder
    :param molecule: rdkit molecule object
    :param fptype: useless here
    :return: latent vector - based feature vector
    """

    smile = Chem.MolToSmiles(molecule, canonical=True)
    smile = Preprocessor.prepare_smiles(smile, SMILES_LEN_ENUM2ENUM)
    if smile is None:
        return [None for x in range(256)]
    smile_vec = Preprocessor.vectorize(smile, CHARSET_ENUM2ENUM, CHAR_TO_INT_ENUM2ENUM, SMILES_LEN_ENUM2ENUM)[0]

    keras.backend.set_session(SMI_TO_LAT_SESSION)

    with SMI_TO_LAT_GRAPH.as_default():
        latent_vec = ENUM2ENUM_SMI_TO_LAT_MODEL.predict(smile_vec)[0]

    return latent_vec


def enum2can(molecule, fptype):
    """
    Function returns a latent vector - based feature vector for a molecule
    which is extracted from enumerated-to-canonical seq2seq SMILES autoencoder
    :param molecule: rdkit molecule object
    :param fptype: useless here
    :return: latent vector - based feature vector
    """

    smile = Chem.MolToSmiles(molecule, canonical=True)
    smile = Preprocessor.prepare_smiles(smile, SMILES_LEN_ENUM2CAN)
    if smile is None:
        return [None for x in range(256)]
    smile_vec = Preprocessor.vectorize(smile, CHARSET_ENUM2CAN, CHAR_TO_INT_ENUM2CAN, SMILES_LEN_ENUM2CAN)[0]

    keras.backend.set_session(SMI_TO_LAT_SESSION)

    with SMI_TO_LAT_GRAPH.as_default():
        latent_vec = ENUM2CAN_SMI_TO_LAT_MODEL.predict(smile_vec)[0]

    return latent_vec

def can2enum(molecule, fptype):
    """
    Function returns a latent vector - based feature vector for a molecule
    which is extracted from canonical-to-enumerated seq2seq SMILES autoencoder
    :param molecule: rdkit molecule object
    :param fptype: useless here
    :return: latent vector - based feature vector
    """

    smile = Chem.MolToSmiles(molecule, canonical=True)
    smile = Preprocessor.prepare_smiles(smile, SMILES_LEN_CAN2ENUM)
    if smile is None:
        return [None for x in range(256)]
    smile_vec = Preprocessor.vectorize(smile, CHARSET_CAN2ENUM, CHAR_TO_INT_CAN2ENUM, SMILES_LEN_CAN2ENUM)[0]

    keras.backend.set_session(SMI_TO_LAT_SESSION)

    with SMI_TO_LAT_GRAPH.as_default():
        latent_vec = CAN2ENUM_SMI_TO_LAT_MODEL.predict(smile_vec)[0]

    return latent_vec

def get_headers(fingerprint, size, diameter):
    name = get_fingerprint_name(fingerprint)
    if diameter:
        header = ['{}{}_{}'.format(name, diameter, i) for i in range(size)]
    else:
        header = ['{}_{}'.format(name, i) for i in range(size)]

    return header


def get_desc_data(molecule):
    """
    Custom function that calculates and returns every available molecular
    descriptor in RDKit chemoinfo toolkt with corresponding header (name) for each
    :param molecule: rdkit's molecule object
    :return: values of descriptors and their headers
    """
    desc_dict = dict(Descriptors.descList)
    descs = list(desc_dict.keys())
    descs.remove('Ipc')
    ans = {}
    for descname in descs:
        try:
            desc = desc_dict[descname]
            bin_value = desc(molecule)
        except (ValueError, TypeError, ZeroDivisionError) as exception:
            print(
                'Descriptor {} wasn\'t calculated for a molecule {} due to {}'.format(
                    str(descname), str(MolToSmiles(molecule)), str(exception))
            )
            bin_value = 'NaN'

        bin_name = 'DESC_{}'.format(descname)
        ans[bin_name] = bin_value

    molstring = np.fromiter(ans.values(), dtype=float)
    headers = np.fromiter(ans.keys(), dtype='S32')

    return molstring, headers

# fingerprints 'object'
# contain name, code, molstring (function), headers (function)
# using to apply fingerprints to some chemical dataset
FINGERPRINTS = {
    # name of each fingerprint
    'name': {
        FCFP: 'FCFP',
        ECFP: 'ECFP',
        FCFC: 'FCFC',
        ECFC: 'ECFC',
        ATOM_PAIRS: 'ATOM_PAIRS',
        MACCS: 'MACCS',
        PATTERN: 'PATTERN',
        AVALON: 'AVALON',
        RDK: 'RDK',
        LAYERED: 'LAYERED',
        ERG: 'EGR',
        PHARMA: 'PHARMA',
        DESC: 'DESC',
        FCFC_CHIRALITY: 'FCFC_CHIRALITY',
        ECFC_CHIRALITY: 'ECFC_CHIRALITY',
        CAN2CAN: 'CAN2CAN',
        ENUM2CAN: 'ENUM2CAN',
        CAN2ENUM: 'CAN2ENUM',
        ENUM2ENUM: 'ENUM2ENUM'
    },
    # code of each fingerprint
    'code': {
        1: FCFP,
        2: ECFP,
        3: FCFC,
        4: ECFC,
        5: ATOM_PAIRS,
        6: MACCS,
        7: PATTERN,
        8: AVALON,
        9: RDK,
        10: LAYERED,
        11: ERG,
        12: PHARMA,
        13: DESC,
        14: FCFC_CHIRALITY,
        15: ECFC_CHIRALITY,
        16: CAN2CAN,
        17: ENUM2CAN,
        18: CAN2ENUM,
        19: ENUM2ENUM
    },
    # methods for get molstring for each fingerprint
    'molstring': {
        FCFP: fcfp_molstring,
        ECFP: ecfp_molstring,
        FCFC: fcfc_molstring,
        ECFC: ecfc_molstring,
        ATOM_PAIRS: atom_pairs_molstring,
        MACCS: maccs_molstring,
        PATTERN: pattern_molstring,
        AVALON: avalon_molstring,
        RDK: rdk_molstring,
        LAYERED: layered_molstring,
        ERG: erg_molstring,
        PHARMA: pharma_molstring,
        FCFC_CHIRALITY: fcfc_chirality_molstring,
        ECFC_CHIRALITY: ecfc_chirality_molstring,
        CAN2CAN: can2can,
        ENUM2CAN: enum2can,
        CAN2ENUM: can2enum,
        ENUM2ENUM: enum2enum
    },
    # methods for get header for each fingerprint
    'headers': {
        FCFP: get_headers,
        ECFP: get_headers,
        FCFC: get_headers,
        ECFC: get_headers,
        ATOM_PAIRS: get_headers,
        MACCS: get_headers,
        PATTERN: get_headers,
        AVALON: get_headers,
        RDK: get_headers,
        LAYERED: get_headers,
        ERG: get_headers,
        PHARMA: get_headers,
        FCFC_CHIRALITY: get_headers,
        ECFC_CHIRALITY: get_headers,
        CAN2CAN: get_headers,
        ENUM2CAN: get_headers,
        CAN2ENUM: get_headers,
        ENUM2ENUM: get_headers
    }
}


def fingerprint_name_by_code(fingerprint_code):
    """
    Method which return fingerprint name by fingerprint number.
    fingerprint_number may contain any 'intable' type

    :param fingerprint_code: number of fingerprint
    :type fingerprint_code: str
    :return: algorithm name or 'Unknown fingerprint'
        if fingerprint number not exist in fingerprint dict
    :rtype: str
    """

    fingerprint_number_as_int = int(fingerprint_code)
    fingerprint_name = 'Unknown fingerprint'
    if fingerprint_number_as_int in FINGERPRINTS['code']:
        fingerprint = FINGERPRINTS['code'][fingerprint_number_as_int]
        fingerprint_name = FINGERPRINTS['name'][fingerprint]

    return fingerprint_name


def fingerprint_type_by_name(name):
    """
    Method which return fingerprint type by its name,
    or None if type does not exist

    :param name: name which want to correspond to type
    :type name: str
    :return: fingerprint type or None
        if fingerprint with given name does not exist
    """

    for fingerprint_type, fingerprint_name in FINGERPRINTS['name'].items():
        if name == fingerprint_name:
            return fingerprint_type

    return None


def get_fingerprint_name(fingerprint):
    return FINGERPRINTS['name'][fingerprint]


def validate_fingerprints(fingerprints):
    """
    Method for validate fingerprints, raise exception id invalid
    Fingerpints should be type of list, with less than 5 entries.
    Each entry is dict, which can have only 'Type', 'Radius' and 'Size' keys.
    Size value can be only int type, and have 0, 256, 512, 1024, 2048 values
    Radius value can be only int type, and have 0, 2, 3, 4 values
    Type value can be only str type, and have one of the FINGERPRINTS['name']

    :param fingerprints: user's input fingerprints
    """

    # check fingerprints type, should be list
    if not isinstance(fingerprints, list):
        raise Exception('User input Fingerprints should be list')
    # check fingerprints count, should be 4 or less
    if not len(fingerprints) <= 4:
        raise Exception(
            'User input Fingerprints should contain 4 or less fingerprints')
    # define possible keys for each fingerprint
    possible_fingerprint_keys = ('Type', 'Radius', 'Size')
    for fingerprint in fingerprints:
        fingerprint_keys = set(fingerprint.keys())
        # check current fingerprint keys
        if not fingerprint_keys.issubset(possible_fingerprint_keys):
            raise Exception(
                'Fingerprint {} have wrong key value'.format(fingerprint))
        # check current fingerprint size
        if 'Size' in fingerprint_keys:
            validate_fingerprint_size(fingerprint)
        # check current fingerprint radius
        if 'Radius' in fingerprint_keys:
            validate_fingerprint_radius(fingerprint)
        # 'Type' MUST be in fingerprint keys
        if 'Type' not in fingerprint_keys:
            raise Exception('Fingerprint {} have not required key Type'.format(
                fingerprint))
        # check current fingerprint type
        validate_fingerprint_type(fingerprint)


def validate_fingerprint_size(fingerprint):
    """
    Method for validate fingerprint size, raise exception id invalid
    Size value can be only int type, and have 0, 256, 512, 1024, 2048 values

    :param fingerprint: fingerprint for validation
    """

    # get fingerprint size value
    size = fingerprint['Size']
    # check fingerprint size type, should be int
    if not isinstance(size, int):
        raise Exception(
            'Fingerprint {} Size should be int'.format(fingerprint)
        )
    # check fingerprint size value, should be 0, 256, 512, 1024 or 2048
    if size not in [0, 128, 256, 512, 1024, 2048]:
        raise Exception(
            'Fingerprint {} Size should be 128, 256, 512, 1024 or 2048'.format(
                fingerprint
            )
        )


def validate_fingerprint_radius(fingerprint):
    """
    Method for validate fingerprint radius, raise exception id invalid
    Radius value can be only int type, and have 0, 2, 3, 4 values

    :param fingerprint: fingerprint for validation
    """

    # get fingerprint radius value
    radius = fingerprint['Radius']
    # check fingerprint radius type, should be int
    if not isinstance(radius, int):
        raise Exception(
            'Fingerprint {} Radius should be int'.format(fingerprint)
        )
    # check fingerprint radius value, should be 0, 2, 3, 4
    if not 2 <= radius <= 4 and radius != 0:
        raise Exception(
            'Fingerprint {} Radius should be in interval [2, 4]'.format(
                fingerprint))


def validate_fingerprint_type(fingerprint):
    """
    Method for validate fingerprint type, raise exception id invalid
    Type value can be only str type, and have one of the FINGERPRINTS['name']
    Change any fingerprint type case to upper case

    :param fingerprint: fingerprint for validation
    """
    # get fingerprint type value
    fingerprint_type = fingerprint['Type']
    # check fingerprint 'Type' type, should be str
    if not isinstance(fingerprint_type, str):
        raise Exception(
            'Fingerprint {} Type should be str'.format(fingerprint)
        )
    # get list of possible fingerprint types
    possible_fingerprints = FINGERPRINTS['name'].values()
    fingerprint_type = fingerprint_type.upper()
    fingerprint['Type'] = fingerprint_type
    # check fingerprint 'Type' value
    if fingerprint_type not in possible_fingerprints:
        raise Exception('Fingerprint {} Type should be: {}'.format(
                fingerprint, possible_fingerprints))


def get_molstring_and_headers(molecule, fptype):
    """
    Function that generates a certain type of fingerprint\descriptors vector for
    a molecule and returns corresponding feature vector and headers
    :param molecule: molecule object
    :param fptype: type, radius and size of fingerprint
    :type fptype: dict
    :return: feature vector and headers
    """
    fingerprint = fingerprint_type_by_name(fptype['Type'])
    if not fingerprint:
        raise ValueError('Unsupported FPtype: {}'.format(fptype['Type']))

    if fingerprint == DESC:
        molstring, headers = get_desc_data(molecule)
    else:
        molstring = FINGERPRINTS['molstring'][fingerprint](molecule, fptype)
        diameter = None
        if 'Radius' in fptype.keys():
            diameter = fptype['Radius'] * 2
        headers = FINGERPRINTS['headers'][fingerprint](
            fingerprint, len(molstring), diameter)

    return molstring, headers
