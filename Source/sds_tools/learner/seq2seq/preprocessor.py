import gc
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Param:
    def __init__(self,charset,char_to_int,int_to_char,smiles_len):
        self.charset = charset
        self.char_to_int = char_to_int
        self.int_to_char = int_to_char
        self.smiles_len= smiles_len


class Preprocessor:

    """ Class is designed to take care of any preprocessing
    of SMILES strings and their representation """

    def __init__(self, filepath, delimiter, smiles_header, smiles_len=70):
        print('Started preprocessing of input data')

        self.smiles_len = smiles_len
        data = pd.read_csv(filepath, delimiter=delimiter)
        data = self.prepare_smiles(data,self.smiles_len,smiles_header=smiles_header)
        self.charset, self.char_to_int, self.int_to_char = \
            Preprocessor.get_charset_charints(data,smiles_header)

        smiles_train, smiles_test = train_test_split(data[smiles_header], random_state=6)
        gc.collect()
        print(self.charset)
        print(smiles_train.shape)
        print(smiles_test.shape)
        print(smiles_test)
        print(smiles_train.iloc[0])



        self.X_train,self.T_train,self.Y_train = Preprocessor.vectorize(
            smiles_train.values,self.charset,self.char_to_int, self.smiles_len
        )
        self.X_test,self.T_test,self.Y_test = Preprocessor.vectorize(
            smiles_test.values,self.charset,self.char_to_int, self.smiles_len
        )

        '''The SMILES must be vectorized to one-hot encoded arrays. 
        To do this a character set is built from all characters found 
        in the SMILES string (both train and test). Also, some start 
        and stop characters are added, which will be used to initiate 
        the decoder and to signal when SMILES generation has stopped. 
        The stop character also work as padding to get the same length 
        of all vectors, so that the network can be trained in batch mode. 
        The character set is used to define two dictionaries to translate back 
        and forth between index and character. The maximum length of the SMILES 
        strings is needed as the RNN’s will be trained in batch mode, and is set 
        to the maximum encountered + some extra.'''

    @staticmethod
    def get_charset_charints(data,smiles_header):
        """
        Analyses characters that are present in the training dataset
        and creates a dictionary to use them later for constant encoding
        :param data:
        :param smiles_header:
        :return:
        """
        charset = set("".join(list(data[smiles_header])) + "!E")
        char_to_int = dict((c, i) for i, c in enumerate(charset))
        int_to_char = dict((i, c) for i, c in enumerate(charset))
        return charset,char_to_int,int_to_char

    @staticmethod
    def prepare_smiles(data,smiles_len,smiles_header=None):
        """
        function which discards unacceptable SMILES
        :param data: SMILES string
        :param smiles_len: max allowed length of the SMILES string
        :return: acceptable SMILES for further modeling
        """
        if type(data) is str:
            smiles = data
            smiles = Preprocessor.fold_double_characters(data)
            if re.compile("[Mga.uUerbLGTfKRmd*h]").search(smiles) or len(smiles) >= smiles_len:
                return None
            else:
                return smiles

        else:
            if smiles_header is None:
                raise Exception('Smiles header is not provided. Cannot locate smiles column in dataframe.')
            else:
                data[smiles_header] = data[smiles_header].apply(Preprocessor.fold_double_characters)
                mask_1 = (~data[smiles_header].str.contains(re.compile("[Mga.uUerbLGTfKRmd*h]")))
                mask_2 = (data[smiles_header].str.len() < smiles_len)

            return data.loc[mask_1].loc[mask_2]

    @staticmethod
    def fold_double_characters(smiles,rep=None):
        """
        method folds double characters in SMILES to special single characters
        :param smiles: SMILES string - text
        :param rep: Dictionary of which double chars fold to which single chars
        :return: SMILES string with folded double chars
        """
        if rep is None:
            rep = {"Cl": "X", "Br": "Y", "Si": "A", "Se": "Z", "se": "z", "As": "Q"}
        if type(smiles) is str:
            # define desired replacements here
            rep = dict((re.escape(k), v) for k, v in iter(rep.items()))
            pattern = re.compile("|".join(rep.keys()))
            smiles = pattern.sub(lambda m: rep[re.escape(m.group(0))], smiles)

            return smiles

        else:
            raise TypeError("Not a string type provided!")

    @staticmethod
    def unfold_double_characters(smiles,rep=None):
        """

        :param smiles:
        :param rep:
        :return:
        """
        if rep is None:
            rep = {"Cl": "X", "Br": "Y", "Si": "A", "Se": "Z", "se": "z", "As": "Q"}
            inv_rep = {v: k for k, v in rep.items()}
        else:
            inv_rep = {v: k for k, v in rep.items()}
        if type(smiles) is str:
              # define desired replacements here
            rep = dict((re.escape(k), v) for k, v in iter(inv_rep.items()))
            pattern = re.compile("|".join(inv_rep.keys()))
            smiles = pattern.sub(lambda m: inv_rep[re.escape(m.group(0))], smiles)

            return smiles

        else:
            raise TypeError("Not a string type provided!")

    @staticmethod
    def vectorize(smiles, charset, char_to_int, smiles_len):
        """
        :param smiles: iterable of smiles-strings
        :param charset:
        :param char_to_int:
        :param smiles_len:
        :return:
        """

        embed = smiles_len + 1

        if type(smiles) is str:
            one_hot = np.zeros((1, embed, len(charset)), dtype=np.int8)
            # encode the startchar
            one_hot[0, 0, char_to_int["!"]] = 1
            # encode the rest of the chars
            for j, c in enumerate(smiles):
                one_hot[0, j + 1, char_to_int[c]] = 1
            # Encode endchar
            one_hot[0, len(smiles) + 1:, char_to_int["E"]] = 1
            # Return two, one for input and the other for output

            return one_hot[:, :0:-1, :], one_hot[:, :-1, :], one_hot[:, 1:, :]

        else:
            one_hot = np.zeros((smiles.shape[0], embed, len(charset)), dtype=np.int8)
            for i, smile in enumerate(smiles):
                # encode the startchar
                one_hot[i, 0, char_to_int["!"]] = 1
                # encode the rest of the chars
                for j, c in enumerate(smile):
                    one_hot[i, j + 1, char_to_int[c]] = 1
                # Encode endchar
                one_hot[i, len(smile) + 1:, char_to_int["E"]] = 1
            # Return two, one for input and the other for output
            return one_hot[:, :0:-1, :],one_hot[:, :-1, :],one_hot[:, 1:, :]
