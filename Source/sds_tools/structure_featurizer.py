import functools
import itertools
import json
from collections import Counter
from itertools import chain

import networkx as nx
import numpy as np
import pandas as pd
# from deepchem.feat.mol_graphs import WeaveMol
from matminer.featurizers.site import AGNIFingerprints, VoronoiFingerprint, CoordinationNumber
from pymatgen import Structure
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import MultiWeightsChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element, Specie, DummySpecie

TOLERANCE_DISTANCE = 0.25

ATOMIC_RADII = {
    'At': 1.50,
    'Bk': 1.70,
    'Cm': 1.74,
    'Fr': 2.60,
    'He': 0.28,
    'Kr': 1.16,
    'Lr': 1.71,
    'Md': 1.94,
    'Ne': 0.58,
    'No': 1.97,
    'Rn': 1.50,
    'Xe': 1.40,
}

CETYPES = [
    'S:1', 'L:2', 'A:2', 'TL:3', 'TY:3', 'TS:3', 'T:4',
    'S:4', 'SY:4', 'SS:4', 'PP:5', 'S:5', 'T:5', 'O:6',
    'T:6', 'PP:6', 'PB:7', 'ST:7', 'ET:7', 'FO:7', 'C:8',
    'SA:8', 'SBT:8', 'TBT:8', 'DD:8', 'DDPN:8', 'HB:8',
    'BO_1:8', 'BO_2:8', 'BO_3:8', 'TC:9', 'TT_1:9',
    'TT_2:9', 'TT_3:9', 'HD:9', 'TI:9', 'SMA:9', 'SS:9',
    'TO_1:9', 'TO_2:9', 'TO_3:9', 'PP:10', 'PA:10',
    'SBSA:10', 'MI:10', 'S:10', 'H:10', 'BS_1:10',
    'BS_2:10', 'TBSA:10', 'PCPA:11', 'H:11', 'SH:11',
    'CO:11', 'DI:11', 'I:12', 'PBP:12', 'TT:12', 'C:12',
    'AC:12', 'SC:12', 'S:12', 'HP:12', 'HA:12', 'SH:13',
    'DD:20'
]

AGNI_MIN = np.zeros((8))
AGNI_MAX = np.array([0.75, 2.0, 6.2, 19.0, 42.0, 65.0, 82.0, 91.0])

VORONOI_MIN = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
VORONOI_MAX = np.array([120.0, 135.0, 11.0, 3.0, 11.0, 12.0, 18.0, 7.0, 17.0, 17.0, 6.0, 2.0, 6.0, 7.0])

COORD_MIN = 1
COORD_MAX = 36

with open('elemental_properties.json') as f:
    PROPERTIES = json.load(f)

UFD = 'universal'  # 'Universal fragment descriptors'
SD = 'structural'  # 'Structural descriptors'
PCD = 'physicochemical'  # 'Physicochemical descriptors'


def get_descriptor_by_name(descriptor_name, structures, parameters):
    DESCRIPTORS = {
        UFD: StructureFeaturizer(**parameters).universal_fingerprints_to_numpy_array(structures),
        SD: StructureFeaturizer(**parameters).get_structural_descriptors(structures),
        PCD: StructureFeaturizer(**parameters).get_elem_properties(structures, ['stats'], ['n_ws_third', 'phi']),
    }

    return DESCRIPTORS[descriptor_name]


def generate_csv(filepath_structures, descriptors, filepath_csv, parameters={}):
    structures = [Structure.from_file(filepath_structure) for filepath_structure in filepath_structures]
    structure_names = [str(structure.formula) for structure in structures]
    d = {'structure_name': structure_names}
    df = pd.DataFrame(data=d)

    for descriptor_name in descriptors:
        fps_array, fps_names = get_descriptor_by_name(descriptor_name, structures, parameters)
        fps_array, fps_names = StructureFeaturizer().remove_nans(fps_array, fps_names)
        df = pd.concat([df, pd.DataFrame(data=fps_array, columns=fps_names)], axis=1)

    df.to_csv(path_or_buf=filepath_csv)

    return df.shape

class StructureFeaturizer():
    def __init__(
            self, tolerance_distance=TOLERANCE_DISTANCE,
            coord_min=COORD_MIN, coord_max=COORD_MAX
    ):
        self.tolerance_distance = tolerance_distance
        self.coord_min = np.array([coord_min])
        self.coord_max = np.array([coord_max])

    def graph_distance_matrix(self, structure):
        adj_mat = self.adjacency_matrix(structure)
        graph = nx.from_numpy_matrix(adj_mat)
        dist_matrix = nx.floyd_warshall_numpy(graph)
        dist_matrix[dist_matrix == np.inf] = dist_matrix[np.isfinite(dist_matrix)].max() + 1
        return np.array(dist_matrix)

    def distance_matrix(self, structure):
        if isinstance(structure, Structure):
            return structure.distance_matrix
        else:
            raise TypeError('structure argument should be pymatgen.Structure')

    def adjacency_matrix(self, structure, fill_diagonal=0):
        dist_matrix = self.distance_matrix(structure)
        adj_matrix = np.zeros(dist_matrix.shape)
        for index, x in np.ndenumerate(dist_matrix):
            specie_1 = structure.sites[index[0]].specie
            specie_2 = structure.sites[index[1]].specie

            if isinstance(specie_1, DummySpecie) and '_symbol' in specie_1.__dict__ and specie_1._symbol == 'D':
                radius_1 = 0.25
            elif isinstance(specie_1, Element):
                radius_1 = specie_1.atomic_radius or ATOMIC_RADII[str(specie_1)]
            else:
                radius_1 = specie_1.element.atomic_radius or ATOMIC_RADII[str(specie_1.element)]

            if isinstance(specie_2, DummySpecie) and '_symbol' in specie_2.__dict__ and specie_2._symbol == 'D':
                radius_2 = 0.25
            elif isinstance(specie_2, Element):
                radius_2 = specie_2.atomic_radius or ATOMIC_RADII[str(specie_2)]
            else:
                radius_2 = specie_2.element.atomic_radius or ATOMIC_RADII[str(specie_2.element)]

            max_distance = radius_1 + radius_2 + self.tolerance_distance
            if x < max_distance:
                adj_matrix[index] = 1
        np.fill_diagonal(adj_matrix, fill_diagonal)
        return adj_matrix

    def overlap_matrix(self, structure):
        dist_matrix = self.distance_matrix(structure)
        overlap_matrix = np.ones(dist_matrix.shape)
        for index, x in np.ndenumerate(dist_matrix):
            radius_1 = Element(structure.sites[index[0]].specie).atomic_radius or ATOMIC_RADII[
                str(structure.sites[index[0]].specie)]
            radius_2 = Element(structure.sites[index[1]].specie).atomic_radius or ATOMIC_RADII[
                str(structure.sites[index[1]].specie)]
            max_distance = radius_1 + radius_2 + self.tolerance_distance
            if x < max_distance:
                overlap_matrix[index] = (x - radius_1 - radius_2) / max_distance
        np.fill_diagonal(overlap_matrix, 0)
        return overlap_matrix

    def _expand_by_zeros(self, matrix, dim_v, dim_h):
        matrix = np.hstack([matrix, np.zeros((matrix.shape[0], dim_h - matrix.shape[1]))])
        matrix = np.vstack([matrix, np.zeros((dim_v - matrix.shape[0], dim_h))])
        return matrix

    def _get_atom_features(self, structure, properties, featurizers):
        def chemenv_fps_for_site(site_index, lse, max_csm=8.0):
            ce_fps = np.zeros(len(CETYPES))
            for ce in lse.coordination_environments[site_index]:
                ce_fps[CETYPES.index(ce['ce_symbol'])] = 1 - ce['csm'] / max_csm
            return ce_fps.tolist()

        atom_features = []
        if 'chemenv' in featurizers:
            lgf = LocalGeometryFinder()
            lgf.setup_parameters(centering_type='centroid', include_central_site_in_centroid=True)
            lgf.setup_structure(structure=structure)
            se = lgf.compute_structure_environments(maximum_distance_factor=1.41, only_cations=False)
            strategy = MultiWeightsChemenvStrategy.stats_article_weights_parameters()
            lse = LightStructureEnvironments.from_structure_environments(strategy=strategy, structure_environments=se)

        for i, site in enumerate(structure.sites):
            atom_feature_vector = []
            for atom_property in properties:
                min_value = np.nanmin(np.array(list(atom_property.values()), dtype=float))
                max_value = np.nanmax(np.array(list(atom_property.values()), dtype=float))
                if atom_property[str(Element(site.specie))] is not None:
                    atom_feature_vector.append(
                        (atom_property[str(Element(site.specie))] - min_value) / (max_value - min_value))
                else:
                    atom_feature_vector.append(None)

            if 'agni' in featurizers:
                agni_fps = (AGNIFingerprints(directions=(None,)).featurize(structure, i) - AGNI_MIN) / (
                    AGNI_MAX - AGNI_MIN)
                atom_feature_vector.extend(agni_fps.tolist())

            if 'voronoi' in featurizers:
                voronoi_fps = VoronoiFingerprint().featurize(structure, i)
                i_fold_symmetry_indices = voronoi_fps[8:16]
                voronoi_stats = (np.array(voronoi_fps[16:]) - VORONOI_MIN) / (VORONOI_MAX - VORONOI_MIN)
                atom_feature_vector.extend(i_fold_symmetry_indices + voronoi_stats.tolist())

            if 'chemenv' in featurizers:
                atom_feature_vector.extend(chemenv_fps_for_site(i, lse))

            if 'coord' in featurizers:
                coord_fps = ((CoordinationNumber.from_preset("MinimumDistanceNN").featurize(
                    structure, i) - self.coord_min) / (self.coord_max - self.coord_min)).tolist()
                atom_feature_vector.extend(coord_fps)

            atom_features.append(atom_feature_vector)

        atom_features = np.array(atom_features, dtype=np.float)

        if np.isnan(atom_features).any():
            raise ValueError('feature vector contains nan value')

        return atom_features

    def _get_bond_features(self, structure):
        feature_matrix = np.zeros((len(structure.sites), len(structure.sites), 4))
        feature_matrix[:, :, 0] = self.adjacency_matrix(structure)
        feature_matrix[:, :, 1] = self.overlap_matrix(structure)
        feature_matrix[:, :, 2] = self.distance_matrix(structure)
        feature_matrix[:, :, 3] = self.graph_distance_matrix(structure)
        return feature_matrix

    def _get_graph_with_labels(self, structure):
        element_labels = {}
        for index, site in enumerate(structure._sites):
            if '_symbol' in site.specie.__dict__ and site.specie._symbol == 'D':
                element_labels[index] = 'D'
            else:
                if isinstance(site.specie, Element):
                    element_labels[index] = site.specie.symbol
                else:
                    element_labels[index] = site.specie.element.symbol
        graph = nx.from_numpy_array(self.adjacency_matrix(structure, fill_diagonal=0))
        nx.set_node_attributes(graph, element_labels, 'element')
        return graph

    def _get_neighbors_dicts(self, graph):
        neighbors_dicts = []
        for node in graph.nodes:
            neighbors = graph.neighbors(node)
            ls = "_".join(sorted([graph.nodes[neighbor]['element'] for neighbor in neighbors]))
            neighbors_dicts.append(str(graph.nodes[node]['element']) + '__' + ls)
        return "____".join(sorted(neighbors_dicts))

    def _get_local_envs(self, graph, all_bonds=False):
        if all_bonds:
            local_envs = [nx.ego_graph(graph, node) for node in graph.nodes()]
        else:
            local_envs = []
            for node in graph.nodes():
                local_env = nx.ego_graph(graph, node)
                remove_edges = []
                for edge in local_env.edges():
                    if node not in edge:
                        remove_edges.append(edge)
                for remove_edge in remove_edges:
                    local_env.remove_edges_from(remove_edges)
                local_envs.append(local_env)
        return local_envs

    def _get_unique_chains(self, graph, cutoff=3, exclude_single_atoms=False):
        all_chains = []
        for df in nx.all_pairs_shortest_path(graph, cutoff=cutoff):
            all_chains.extend(list(df[1].values()))
        if exclude_single_atoms:
            unique_chains = [graph.subgraph(el) for el in all_chains if el[0] <= el[-1] and len(el) > 1]
        else:
            unique_chains = [graph.subgraph(el) for el in all_chains if el[0] <= el[-1]]
        return unique_chains

    def _get_unique_fragments(self, graphs, init_fragments):
        all_fragments = init_fragments
        neighbors_dicts = [self._get_neighbors_dicts(graph) for graph in graphs]
        new_fragments = dict(zip(Counter(neighbors_dicts).keys(), Counter(neighbors_dicts).values()))
        for new_fragment_key in new_fragments.keys():
            if new_fragment_key not in all_fragments.keys():
                all_fragments[new_fragment_key] = new_fragments[new_fragment_key]
        return all_fragments

    def generate_weavemol_from_structure(self, structure, properties, featurizers):
        try:
            return WeaveMol(self._get_atom_features(structure, properties, featurizers),
                            self._get_bond_features(structure))
        except TypeError:
            return None

    def generate_convmol_from_structure(self, structure, properties, featurizers, max_n_atoms=200, n_features=50):
        try:
            adj_matrix = self._expand_by_zeros(self.adjacency_matrix(structure), max_n_atoms, max_n_atoms)
            atom_features = self._expand_by_zeros(self._get_atom_features(structure, properties, featurizers),
                                                  max_n_atoms, n_features)
            return [adj_matrix, atom_features, len(structure.sites)]
        except TypeError:
            return None

    def _structure_preprocessing(self, structure, supercell_coefficient):
        is_ordered = []
        structure = structure.__mul__(supercell_coefficient)

        for site in structure.sites:
            is_ordered.append(site.is_ordered)

        if False in is_ordered:
            remove_sites_index = []
            for index, site in enumerate(structure.sites):
                if site.is_ordered is not True:
                    species_and_occu = site.species_and_occu.as_dict()
                    if sum(species_and_occu.values()) < 1:
                        species_and_occu['vacancy'] = 1 - sum(species_and_occu.values())
                    specie = np.random.choice(
                        [x for x in species_and_occu for y in range(int(100 * species_and_occu[x]))])
                    if specie == 'vacancy':
                        remove_sites_index.append(index)
                    else:
                        if specie == 'D+':
                            structure[index] = (Composition({specie: sum(species_and_occu.values())}), site.coords)
                        else:
                            try:
                                structure[index] = Specie.from_string(specie)
                            except ValueError:
                                structure[index] = Specie(specie)
            structure.remove_sites(remove_sites_index)

        return structure

    def structure_preprocessing(self, structures, supercell_coefficient=1):
        flat_structures = list(map(functools.partial(
            self._structure_preprocessing,
            supercell_coefficient=supercell_coefficient), structures))
        return flat_structures

    def _generate_universal_fragment_fingerprints(self, structure, init_fragments=None,
                                                  multiple_coefficient=1, supercell_coefficient=1,
                                                  cutoff=3, all_bonds=False, only_bonds=False):
        sum_up = []
        if init_fragments is None:
            init_fragments = {}
        cutoff = 1 if only_bonds else cutoff
        for _ in range(multiple_coefficient):
            flat_structure = self._structure_preprocessing(
                structure, supercell_coefficient=supercell_coefficient)
            graph_with_labels = self._get_graph_with_labels(flat_structure)
            unique_chains = self._get_unique_fragments(
                self._get_unique_chains(
                    graph_with_labels, cutoff=cutoff, exclude_single_atoms=only_bonds), init_fragments)
            local_envs = {} if only_bonds else self._get_local_envs(graph_with_labels, all_bonds)
            unique_chains_and_local_envs = self._get_unique_fragments(local_envs, unique_chains)
            sum_up.append(Counter(unique_chains_and_local_envs))

        sum_dict = dict(sum(sum_up, Counter()))
        bonds_sum = sum(sum_dict.values()) if only_bonds else 1
        sum_dict.update((key, value / (multiple_coefficient * bonds_sum)) for key, value in sum_dict.items())
        return sum_dict

    def featurize_dataset_with_universal_fragment_fingerprints(self, dataset, multiple_coefficient=1,
                                                               supercell_coefficient=1, cutoff=3,
                                                               all_bonds=False):
        fps = []

        for i, structure in enumerate(dataset):
            fps.append(
                self._generate_universal_fragment_fingerprints(
                    structure, multiple_coefficient=multiple_coefficient,
                    supercell_coefficient=supercell_coefficient,
                    cutoff=cutoff, all_bonds=all_bonds, ))

            if (i + 1) % 100 == 0:
                print('featurizing {0} structures'.format(i + 1))

        return fps

    def featurize_dataset_with_bonds(self, dataset, multiple_coefficient=1, supercell_coefficient=1):
        return list(map(functools.partial(self._generate_bonds, multiple_coefficient=multiple_coefficient,
                                          supercell_coefficient=supercell_coefficient), dataset))

    def _generate_bonds(self, structure, multiple_coefficient=1,
                        supercell_coefficient=1, init_fragments=None):
        if init_fragments is None: init_fragments = {}
        bonds = self._generate_universal_fragment_fingerprints(
            structure, init_fragments=init_fragments,
            multiple_coefficient=multiple_coefficient,
            supercell_coefficient=supercell_coefficient,
            only_bonds=True)
        bonds_sum = sum(bonds.values())
        for key, bond in bonds.items():
            bonds[key] = bond / bonds_sum
        return bonds

    def featurize_dataset_with_graphs(self, dataset, properties, str_featurizers, graph_type='convmol'):
        if graph_type == 'convmol':
            g_featurizer = self.generate_convmol_from_structure
        elif graph_type == 'weavemol':
            g_featurizer = self.generate_weavemol_from_structure
        else:
            raise ValueError('{} is not currently supported graph type'.format(graph_type))
        return list(map(functools.partial(g_featurizer, properties=properties, featurizers=str_featurizers), dataset))

    def universal_fingerprints_to_numpy_array(self, structures, filtering=False, min_fraction=5):
        fps_dicts = self.featurize_dataset_with_universal_fragment_fingerprints(structures)
        numpy_array = []
        all_universal_fingerprints = set(chain.from_iterable(list(map(list, fps_dicts))))
        all_universal_fingerprints = list(all_universal_fingerprints)
        for universal_fingerprints_in_structure in fps_dicts:
            sub_string = []
            for universal_fingerprint in all_universal_fingerprints:
                if universal_fingerprint in universal_fingerprints_in_structure.keys():
                    sub_string.append(universal_fingerprints_in_structure[universal_fingerprint])
                else:
                    sub_string.append(0)
            numpy_array.append(sub_string)
        numpy_array = np.array(numpy_array)

        if filtering is True:
            if min_fraction < 1:
                size = min_fraction * numpy_array.shape[0]
            else:
                size = min_fraction
            numpy_array_transpose = numpy_array.transpose()
            list_of_lost_indexes = []
            for index, string in enumerate(numpy_array_transpose):
                if np.count_nonzero(string) < size:
                    list_of_lost_indexes.append(index)
            list_of_indexes = sorted(set(range(numpy_array.shape[1])) - set(list_of_lost_indexes))
            all_universal_fingerprints = [all_universal_fingerprints[i] for i in list_of_indexes]
            numpy_array_transpose = np.delete(numpy_array_transpose, list_of_lost_indexes, 0)
            numpy_array = numpy_array_transpose.transpose()

        return numpy_array, all_universal_fingerprints

    def _get_site_properties(self, site, properties):
        return [site_property[site.specie.number - 1] for site_property in list(properties.values())]

    def _get_elem_properties(self, structure, excluded_properties):
        properties = {k: v for k, v in PROPERTIES.items() if k not in excluded_properties}
        keys = list(properties.keys())
        values = np.array(list(map(lambda p: self._get_site_properties(p, properties=properties), structure.sites)))
        return values, keys

    def _get_stats(self, prop_dicts):
        properties = prop_dicts[0]
        prop_names = prop_dicts[1]
        stat_names = []

        max_ = np.amax(properties, axis=0)
        min_ = np.amin(properties, axis=0)
        range_ = np.ptp(properties, axis=0)
        median_ = np.median(properties, axis=0)
        average_ = np.average(properties, axis=0)
        mean_ = np.mean(properties, axis=0)
        std_ = np.std(properties, axis=0)
        var_ = np.var(properties, axis=0)

        stat_names.extend(['max ({0})'.format(prop_name) for prop_name in prop_names])
        stat_names.extend(['min ({0})'.format(prop_name) for prop_name in prop_names])
        stat_names.extend(['range ({0})'.format(prop_name) for prop_name in prop_names])
        stat_names.extend(['median ({0})'.format(prop_name) for prop_name in prop_names])
        stat_names.extend(['average ({0})'.format(prop_name) for prop_name in prop_names])
        stat_names.extend(['mean ({0})'.format(prop_name) for prop_name in prop_names])
        stat_names.extend(['std ({0})'.format(prop_name) for prop_name in prop_names])
        stat_names.extend(['var ({0})'.format(prop_name) for prop_name in prop_names])

        stats = np.hstack((max_, min_, range_, median_, average_, mean_, std_, var_))
        return stats.tolist(), stat_names

    def _get_combo_properties(self, prop_dicts):
        combo_props = []
        elem_properties = prop_dicts[0]
        prop_names = prop_dicts[1]

        for property_ in (elem_properties):
            ratio_1 = ([property_[c[0]] / property_[c[1]] for c in itertools.combinations(
                range(elem_properties.shape[1]), 2)])

            ratio_1_names = (['{0} / {1}'.format(prop_names[c[0]], prop_names[c[1]])
                              for c in itertools.combinations(
                    range(elem_properties.shape[1]), 2)])
            ratio_2 = ([property_[c[1]] / property_[c[0]] for c in itertools.combinations(
                range(elem_properties.shape[1]), 2)])
            ratio_2_names = (['{1} / {0}'.format(prop_names[c[0]], prop_names[c[1]])
                              for c in itertools.combinations(
                    range(elem_properties.shape[1]), 2)])
            multy = ([property_[c[1]] * property_[c[0]] for c in itertools.combinations(
                range(elem_properties.shape[1]), 2)])
            multy_names = (['{0} * {1}'.format(prop_names[c[0]], prop_names[c[1]])
                            for c in itertools.combinations(
                    range(elem_properties.shape[1]), 2)])
            all_combos = np.hstack((ratio_1, ratio_2, multy))
            all_names = ratio_1_names + ratio_2_names + multy_names
            combo_props.append(all_combos)
        return np.vstack(combo_props).tolist(), all_names

    def _get_element_properties_for_structure(self, structure, property_types, excluded_properties):
        reciprocal_square_distance_matrix = 1 / np.multiply(
            self.distance_matrix(structure), self.distance_matrix(structure))
        reciprocal_square_distance_matrix[reciprocal_square_distance_matrix == np.inf] = 0
        galvez_matrix = self.adjacency_matrix(structure) @ reciprocal_square_distance_matrix

        stat_props = []
        stat_names = []
        combo_props = []
        combo_names = []
        q_bond = []
        q_all = []
        q_bond_names = []
        q_all_names = []

        if 'stats' in property_types:
            stat_props, stat_names = self._get_stats(
                self._get_elem_properties(
                    structure, excluded_properties=excluded_properties))

        if 'combo' in property_types:
            combo_props, combo_names = self._get_stats(self._get_combo_properties(
                self._get_elem_properties(
                    structure, excluded_properties=excluded_properties)))

        if 'galvez_bond' in property_types:
            bond_combos = np.transpose(np.vstack(np.nonzero(self.adjacency_matrix(structure))))
            bond_prop, bond_names = self._get_elem_properties(structure, excluded_properties=excluded_properties)
            for p, n in zip(np.transpose(bond_prop), bond_names):
                q_bond.append(sum([galvez_matrix[c[0]][c[1]] * abs(p[c[0]] - p[c[1]]) for c in bond_combos]))
                q_bond_names.append('galvez_bond ' + n)

        if 'galvez_all' in property_types:
            g_all_prop, g_all_names = self._get_elem_properties(structure, excluded_properties=excluded_properties)
            for p, n in zip(np.transpose(g_all_prop), g_all_names):
                combinations = itertools.combinations(range(len(structure.sites)), 2)
                q_all.append(sum([galvez_matrix[c[0]][c[1]] * abs(p[c[0]] - p[c[1]]) for c in combinations]))
                q_all_names.append('galvez_all ' + n)

        all_props = stat_props + combo_props + q_bond + q_all
        all_names = stat_names + combo_names + q_bond_names + q_all_names

        return all_props, all_names

    def get_elem_properties(self, structures, property_types, excluded_properties):
        properties = []

        for i, structure in enumerate(structures):
            properties.append(self._get_element_properties_for_structure(
                structure, property_types, excluded_properties)[0])

            if i == 0:
                names = self._get_element_properties_for_structure(
                    structure, property_types, excluded_properties)[1]

            if (i + 1) % 100 == 0:
                print('featurizing {0} structures'.format(i + 1))

        return np.array(properties), names

    def remove_nans(self, values, keys):
        props = values[:, np.all(np.isfinite(values), axis=0)]
        names = np.array(keys)[np.all(np.isfinite(values), axis=0)].tolist()
        return props, names

    def _get_structural_descriptors(self, structure):
        structural_descriptors = {
            'a': structure.lattice.a,
            'b': structure.lattice.b,
            'c': structure.lattice.c,
            'a / b': structure.lattice.a / structure.lattice.b,
            'b / c': structure.lattice.b / structure.lattice.c,
            'a / c': structure.lattice.a / structure.lattice.c,
            'angle_1': structure.lattice.angles[0],
            'angle_2': structure.lattice.angles[1],
            'angle_3': structure.lattice.angles[2],
            'volume': structure.lattice.volume,
            'number_of_species': structure.ntypesp,
            'number_of_atoms': structure.num_sites,
            'volume_per_atom': structure.lattice.volume / structure.num_sites,
        }

        return list(structural_descriptors.values()), list(structural_descriptors.keys())

    def get_structural_descriptors(self, structures):
        fps, names = [], []
        for i, structure in enumerate(structures):
            fps.append(self._get_structural_descriptors(structure)[0])
            names.append(self._get_structural_descriptors(structure)[1])
            if (i + 1) % 100 == 0:
                print('featurizing {0} structures'.format(i + 1))
        fps = np.array(fps)
        return fps, names[0]