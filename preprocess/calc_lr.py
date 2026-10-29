import json
import os
import networkx as nx
import numpy as np
import obonet
from tqdm import tqdm

from config import ANNOTATION_FILE_PATH, LR_PRIOR_PATH, HP_OBO_PATH, PROPAGATED_FREQ_PATH


class DiseaseAnalyzer:
    def __init__(self, graph):
        self.graph = graph
        self.PHENOTYPIC_ABNORMALITY = "HP:0000118"
        if isinstance(self.graph, nx.MultiDiGraph):
            self.graph = nx.DiGraph(self.graph)

    def get_induced_graph(self, annotations, default_f=1e-5):
        term_2_max_frequency = {}
        term_2_prob_inv = {}
        direct_terms = set(annotations.keys())

        for tid, f in annotations.items():
            if tid not in self.graph:
                continue

            ancestor_distances = nx.single_source_shortest_path_length(self.graph, tid)

            for anc_id, dist in ancestor_distances.items():
                if f > term_2_max_frequency.get(anc_id, default_f):
                    term_2_max_frequency[anc_id] = f

                if anc_id not in term_2_prob_inv:
                    term_2_prob_inv[anc_id] = 1.0
                term_2_prob_inv[anc_id] *= (1.0 - f)

        term_2_prob = {
            tid: 1.0 - p_inv if tid not in direct_terms else term_2_max_frequency[tid]
            for tid, p_inv in term_2_prob_inv.items()
        }

        for tid, f in annotations.items():
            if tid in term_2_prob and term_2_prob[tid] < f:
                term_2_prob[tid] = f
            if tid in term_2_max_frequency and term_2_max_frequency[tid] < f:
                term_2_max_frequency[tid] = f

        return {
            "annotations": annotations,
            "max_frequency": term_2_max_frequency,
            "probability": term_2_prob
        }


def calculate_hpo_overall_prob(ontology_graph, induced_disease_graph_map):
    all_terms = list(ontology_graph.nodes())
    N = len(induced_disease_graph_map)
    term_to_idx = {tid: i for i, tid in enumerate(all_terms)}
    prob_vector = np.zeros(len(all_terms))

    for idg in induced_disease_graph_map.values():
        probs = idg.get('probability', {})
        for tid, p in probs.items():
            if tid in term_to_idx:
                prob_vector[term_to_idx[tid]] += p

    final_probs = prob_vector / N if N > 0 else prob_vector
    return dict(zip(all_terms, final_probs))


def _dfs_attenuate_lr(hpo_lrs, cur_id, cur_lr, ontology_graph, hpo_term_2_overall_prob):
    DEFAULT_BACKGROUND_PROBABILITY = 1.0 / 10000
    child_terms = list(ontology_graph.predecessors(cur_id))

    if not child_terms:
        return

    child_prob_sum = sum(
        hpo_term_2_overall_prob.get(child, DEFAULT_BACKGROUND_PROBABILITY)
        for child in child_terms
    )

    if child_prob_sum == 0:
        return

    for child in child_terms:
        child_bg_prob = hpo_term_2_overall_prob.get(child, DEFAULT_BACKGROUND_PROBABILITY)
        child_lr = 0.2 * cur_lr * child_bg_prob / child_prob_sum

        if child_lr > 0.01:
            if child not in hpo_lrs or child_lr > hpo_lrs[child]:
                hpo_lrs[child] = child_lr
                _dfs_attenuate_lr(hpo_lrs, child, child_lr, ontology_graph, hpo_term_2_overall_prob)


def calculate_hpo_2_disease_lr_prob(ontology_graph, induced_disease_graph_map, hpo_term_2_overall_prob):
    DEFAULT_BACKGROUND_PROBABILITY = 1.0 / 10000
    disease_2_hpo_lr = {}

    for disease_id, idg in tqdm(induced_disease_graph_map.items(), 'Calculating LR'):
        hpo_lr_prob = {}

        for hpo_id, freq in idg['annotations'].items():
            background_prob = hpo_term_2_overall_prob.get(hpo_id, DEFAULT_BACKGROUND_PROBABILITY)
            lr = freq / background_prob
            hpo_lr_prob[hpo_id] = lr

        for hpo_id, prob in idg['probability'].items():
            if hpo_id not in hpo_lr_prob:
                background_prob = hpo_term_2_overall_prob.get(hpo_id, DEFAULT_BACKGROUND_PROBABILITY)
                lr = prob / background_prob
                hpo_lr_prob[hpo_id] = lr

        current_keys = list(idg['probability'].keys())
        for hpo_id in current_keys:
            _dfs_attenuate_lr(
                hpo_lr_prob,
                hpo_id,
                hpo_lr_prob[hpo_id],
                ontology_graph,
                hpo_term_2_overall_prob
            )

        disease_2_hpo_lr[disease_id] = hpo_lr_prob

    return disease_2_hpo_lr


def process_all_diseases(json_path=ANNOTATION_FILE_PATH, obo_path=HP_OBO_PATH, output_path_1=PROPAGATED_FREQ_PATH, output_path_2=LR_PRIOR_PATH):
    hpo_graph = obonet.read_obo(obo_path)
    analyzer = DiseaseAnalyzer(hpo_graph)

    with open(json_path, 'r', encoding='utf-8') as f:
        disease_data = json.load(f)

    disease_graph_map = {}
    for orpha_id, annotations in tqdm(disease_data.items(), 'Calculating propagated frequency'):
        disease_graph_map[orpha_id] = analyzer.get_induced_graph(annotations)

    with open(output_path_1, 'w', encoding='utf-8') as f:
        json.dump({k: v['probability'] for k, v in disease_graph_map.items()}, f, indent=2)

    hpo_term_2_overall_prob = calculate_hpo_overall_prob(hpo_graph, disease_graph_map)
    disease_2_hpo_lr = calculate_hpo_2_disease_lr_prob(hpo_graph, disease_graph_map, hpo_term_2_overall_prob)

    with open(output_path_2, 'w', encoding='utf-8') as f:
        json.dump(disease_2_hpo_lr, f, indent=2)


if __name__ == "__main__":
    process_all_diseases()