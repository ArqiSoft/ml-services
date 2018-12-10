"""
MODULE DOCSTRING THERE!!!
"""
# TODO write module docstring
import gzip

import numpy
from rdkit import Chem

from MLLogger import BaseMLLogger
from learner.fingerprints import get_molstring_and_headers
from general_helper import NUMPY_PROCESSOR_DTYPES

LOGGER = BaseMLLogger(log_name='sds_ml_processor_logger')


def sdf_to_csv(
        infile, fptype, write_sdf=False, find_classes=False, find_values=False,
        value_name_list=None, class_name_list=None, units_prop=None,
        cut_off=None, relation_prop=None, stream=None, molecules=None,
        processing_errors=None
):
    """
    This script is designed to simplify data preparation for ML-methods.

    :param fptype:
    :param infile: input sdf\sdf.gz file containing dataset (filename\path)
    :param write_sdf: write processed sdf file
    :param find_classes: enable auto classes detection
        (WARNING!: works only with true\false classes)
    :param find_values: enable auto values detection
        (WARNING!: careful with this one, any property that can be
        floated will be added to values)
    :param value_name_list: desired values names
        (coma separated. Ex: Value1,Value2)
    :param class_name_list: same thing with classes
        (coma separated. Ex: Class1,Class2)
    :param units_prop: name of the Property field containing values units
    :param relation_prop: name of the Property field containing <,>,=
    :param cut_off: adds property 'cut_off_activity'
        comparing cut_off and value(1 if values < cut_off)
        this feature only works if you set 1 value
    :param stream:
    :param molecules:
    :param processing_errors:
    :type infile: string
    :type find_classes: bool
    :type find_values: bool
    :type value_name_list: string
    :type class_name_list: string
    :type units_prop: string
    :type relation_prop: string
    :type cut_off: int
    :return: pandas dataframe 0,0,1,0...,Value1,Values2,Class1,cut_off_activity
        for every molecule
    """

    if not value_name_list and not class_name_list:
        value_name_list = class_name_list = []

    mols = read_molecules(infile=infile, stream=stream, molecules=molecules)

    classes_names = make_classes_names(
        mols, class_name_list=class_name_list, find_classes=find_classes)
    values_names = make_values_names(
        mols, value_name_list=value_name_list, find_values=find_values,
        units_prop=units_prop, cut_off=cut_off, relation_prop=relation_prop,
        processing_errors=processing_errors
    )
    values_names = [x for x in values_names if x not in classes_names]

    if cut_off:
        classes_names.append('cut_off_activity')
        values_names = []

    data_frame = molstrings_and_headers(
        mols, fptype, values_names, classes_names)

    if write_sdf:
        input_no_ext = infile.split('.')[0]
        chem_writer = Chem.SDWriter('{}_cured.sdf'.format(input_no_ext))
        for mol in mols:
            chem_writer.write(mol)

    return data_frame


def read_molecules(infile=None, stream=None, molecules=None):
    if stream:
        suppl = Chem.ForwardSDMolSupplier(stream)
        mols = [x for x in suppl if x is not None]
    elif isinstance(molecules, list):
        mols = molecules
    else:
        if infile.endswith('.sdf.gz'):
            suppl = Chem.ForwardSDMolSupplier(gzip.open(infile))
        elif infile.endswith('.sdf'):
            suppl = Chem.SDMolSupplier(infile)
        else:
            print('Wrong Format!')
            return 1
        mols = [x for x in suppl if x is not None]

    LOGGER.info('{} valid molecules in {} dataset'.format(len(mols), infile))

    return mols


def make_classes_names(mols, class_name_list=None, find_classes=None):
    if class_name_list:
        classes_names = class_name_list.split(',')
        for mol in mols:
            for name in classes_names:
                if mol.GetProp(name).upper() == 'TRUE':
                    mol.SetProp(name, '1')
                elif mol.GetProp(name).upper() == 'FALSE':
                    mol.SetProp(name, '0')
    else:
        if not find_classes:
            classes_names = []
        else:
            classes = set()
            for mol in mols:
                # Can be optimized using GetPropsDict method
                for prop_name in mol.GetPropNames():
                    if mol.GetProp(prop_name).upper() == 'TRUE':
                        classes.add(prop_name)
                        mol.SetProp(prop_name, '1')
                    elif mol.GetProp(prop_name).upper() == 'FALSE':
                        classes.add(prop_name)
                        mol.SetProp(prop_name, '0')

            classes_names = list(classes)

    return classes_names


def make_values_names(
        mols, value_name_list=None, find_values=None, units_prop=None,
        cut_off=None, relation_prop=None, processing_errors=None
):
    errors_list = list()

    if value_name_list:
        values_names = value_name_list.split(',')
        for molecule_number, mol in enumerate(mols):
            for name in values_names:
                try:
                    val = float(mol.GetProp(name))
                    if units_prop:
                        if mol.GetProp(units_prop) == 'mM':
                            val *= 1000
                        elif mol.GetProp(units_prop) == 'nM':
                            pass
                        else:
                            mol.SetProp(name, 'None')
                            continue
                    if cut_off and len(values_names) == 1:
                        if not relation_prop:
                            if val <= cut_off:
                                mol.SetProp('cut_off_activity', '1')
                            else:
                                mol.SetProp('cut_off_activity', '0')
                        else:
                            relation = mol.GetProp(relation_prop)
                            if val == cut_off:
                                if relation == '=' or relation == '<':
                                    mol.SetProp('cut_off_activity', '1')
                                elif relation == '>':
                                    mol.SetProp('cut_off_activity', '0')
                                else:
                                    mol.SetProp('cut_off_activity', 'None')
                            elif val < cut_off:
                                if relation == '>':
                                    mol.SetProp('cut_off_activity', 'None')
                                elif relation == '=' or relation == '<':
                                    mol.SetProp('cut_off_activity', '1')
                                else:
                                    mol.SetProp('cut_off_activity', 'None')
                            elif val > cut_off:
                                if relation == '=' or relation == '>':
                                    mol.SetProp('cut_off_activity', '0')
                                elif relation == '<':
                                    mol.SetProp('cut_off_activity', 'None')
                                else:
                                    mol.SetProp('cut_off_activity', 'None')

                except ValueError:
                    mol.SetProp(name, 'None')
                    errors_list.append(
                        ('Processing error', molecule_number, 'Value error'))
                except KeyError:
                    mol.SetProp(name, 'None')
                    errors_list.append(
                        ('Processing error', molecule_number, 'Value error'))

            errors_list.append(
                ('Processing error', molecule_number, None))

    else:
        if not find_values:
            values_names = []
        else:
            values = set()
            for molecule_number, mol in enumerate(mols):
                for prop_name in mol.GetPropNames():
                    try:
                        float(mol.GetProp(prop_name))
                        values.add(prop_name)
                    except ValueError:
                        mol.SetProp(prop_name, 'None')
                        errors_list.append((
                            'Processing error', molecule_number, 'Value error'
                        ))

                errors_list.append(
                    ('Processing error', molecule_number, None))

            values_names = list(values)

    if processing_errors is not None:
        processing_errors[0] = errors_list

    return values_names


def molstrings_and_headers(
        mols, fptype, values_names, classes_names, write_sdf=None, infile=None
):
    alpha_list = []
    for mol in mols:

        if isinstance(fptype, list):
            for i in range(len(fptype)):
                if i == 0:
                    molstring, headers = get_molstring_and_headers(
                        mol, fptype[i])
                else:
                    molstring_next, headers_next = get_molstring_and_headers(
                        mol, fptype[i])
                    molstring = numpy.concatenate(
                        [molstring, molstring_next], axis=0)
                    headers = numpy.concatenate([headers, headers_next])

            for name in values_names:
                headers = numpy.append(headers, name)
            for name in classes_names:
                headers = numpy.append(headers, name)

        else:
            raise ValueError(
                'Argument fptype must a list of dictionaries with parameters for FP')

        for name in values_names:
            try:
                if type(molstring) is numpy.ndarray:
                    molstring = numpy.append(molstring, float(mol.GetProp(name)))
                else:
                    molstring.append(float(mol.GetProp(name)))
            except (KeyError, ValueError):
                mol.SetProp(name, 'None')
                if type(molstring) is numpy.ndarray:
                    molstring = numpy.append(molstring, numpy.nan)
                else:
                    molstring.append(None)
        for name in classes_names:
            try:
                if type(molstring) is numpy.ndarray:
                    molstring = numpy.append(molstring, int(mol.GetProp(name)))
                else:
                    molstring.append(int(mol.GetProp(name)))
            except (KeyError, ValueError):
                mol.SetProp(name, 'None')
                if type(molstring) is numpy.ndarray:
                    molstring = numpy.append(molstring, 'None')
                else:
                    molstring.append('None')
        # TODO temporary "listification" there
        alpha_list.append(list(molstring))

    TMP_all_data = list()
    for molecule_number, TMP_molecule_row in enumerate(alpha_list):
        TMP_formatted_molecule_row = list()
        for bin_index, molecule_value in enumerate(TMP_molecule_row):
            TMP_tuple = (headers[bin_index], molecule_number, molecule_value)
            TMP_formatted_molecule_row.append(TMP_tuple)

        TMP_all_data.append(TMP_formatted_molecule_row)

    TMP_numpy_array = numpy.array(
        TMP_all_data,
        dtype=NUMPY_PROCESSOR_DTYPES
    )

    if write_sdf:
        input_no_ext = infile.split('.')[0]
        chem_writer = Chem.SDWriter('{}_cured.sdf'.format(input_no_ext))
        for mol in mols:
            chem_writer.write(mol)

    return TMP_numpy_array
