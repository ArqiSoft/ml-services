import numpy as np
import pandas as pd
from pymatgen.ext.matproj import MPRester
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.dos import Dos
from pymatgen.electronic_structure.core import OrbitalType
from pymatgen.electronic_structure.plotter import BSPlotter, DosPlotter
from tsfresh.feature_extraction import extract_features, ComprehensiveFCParameters

FC_PARAMETERS = {
    'absolute_sum_of_changes': None,
    'first_location_of_maximum': None,
    'first_location_of_minimum': None,
    'kurtosis': None,
    'last_location_of_maximum': None,
    'last_location_of_minimum': None,
    'longest_strike_above_mean': None,
    'longest_strike_below_mean': None,
    'maximum': None,
    'mean': None,
    'mean_abs_change': None,
    'mean_change': None,
    'mean_second_derivative_central': None,
    'median': None,
    'minimum': None,
    'sample_entropy': None,
    'skewness': None,
    'standard_deviation': None,
    'variance': None,
}


class BandStructureFeaturizer():
    def _get_bands_as_df(self, bandstructure, spin, n_valence_bands, n_conduction_bands):
        bands = bandstructure.bands[spin] - bandstructure.efermi
        sorted_bands = bands[np.argsort(bands[:, 0])]
        first_cond_band_index = np.argmin(sorted_bands[:, 0] < 0)
        valence_bands = sorted_bands[first_cond_band_index - n_valence_bands: first_cond_band_index]
        conduction_bands = sorted_bands[first_cond_band_index: first_cond_band_index + n_conduction_bands]
        bins = np.vstack((np.zeros(valence_bands.shape[1]), np.linspace(0, 1, valence_bands.shape[1])))
        all_bands = np.vstack((bins, valence_bands, conduction_bands))
        columns = ['id', 'k']

        for num in range(n_valence_bands, 0, -1):
            columns.append(f'{num}_val')
        for num in range(1, n_conduction_bands + 1):
            columns.append(f'{num}_cond')

        return pd.DataFrame(all_bands.transpose(), columns=columns)

    def get_series_features(self, bandstructure, spin=Spin.up, n_valence_bands=5, n_conduction_bands=5):
        bands_as_df = self._get_bands_as_df(bandstructure,
                                            spin=spin,
                                            n_valence_bands=n_valence_bands,
                                            n_conduction_bands=n_conduction_bands)
        features = extract_features(
            bands_as_df, default_fc_parameters=FC_PARAMETERS, column_id="id", column_sort="k")
        return features.values[0], features.columns.tolist()

    def get_b_fingerprints(self, bandstructure, spin=Spin.up, e_min=-20, e_max=20, n_bins=20):
        bands = bandstructure.bands[spin] - bs.efermi
        sorted_bands = bands[np.argsort(bands[:, 0])]
        g_point_energies = sorted_bands[:, 0]
        counts, bins = np.histogram(g_point_energies, bins=np.linspace(e_min, e_max, n_bins + 1))
        return counts, bins


class DoSFeaturizer():
    def _get_spd_dos_as_df(self, dos, orbital_type, e_min, e_max):
        spd_dos = dos.get_spd_dos()[getattr(OrbitalType, orbital_type)]
        energies = spd_dos.energies[((spd_dos.energies > e_min) & (spd_dos.energies < e_max))]
        densities = spd_dos.get_densities()[np.where((spd_dos.energies > e_min) & (spd_dos.energies < e_max))[0]]
        bins = np.vstack((np.zeros(energies.shape[0]), energies))
        all_densities = np.vstack((bins, densities))
        columns = ['id', 'k']
        columns.append(f'{orbital_type}_dos')
        return pd.DataFrame(all_densities.transpose(), columns=columns)

    def _get_total_dos_as_df(self, dos, e_min, e_max, spin):
        energies = dos.energies - dos.efermi
        densities = dos.densities[spin]
        bins = np.vstack((np.zeros(energies.shape[0]), energies))
        total_densities = np.vstack((bins, densities))
        columns = ['id', 'k']
        columns.append('total_dos')
        return pd.DataFrame(total_densities.transpose(), columns=columns)

    def get_series_features(self, dos, dos_type='total', orbital_type='s', e_min=-5.0, e_max=5, spin=Spin.up):
        if dos_type == 'total':
            dos_as_df = self._get_total_dos_as_df(dos, e_min=e_min, e_max=e_max, spin=spin)
        elif dos_type == 'spd':
            dos_as_df = self._get_spd_dos_as_df(dos, orbital_type=orbital_type, e_min=e_min, e_max=e_max)
        else:
            raise ValueError('unknown dos type')
        features = extract_features(
            dos_as_df, default_fc_parameters=FC_PARAMETERS, column_id="id", column_sort="k")
        return features.values[0], features.columns.tolist()