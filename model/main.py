import random

import numpy as np
import torch
import torch.nn as nn

from model.output import ResidualNetworkClassifier

import networkx as nx

from utils.utils import *
from config import *

torch.set_default_device(device)


class DiagDL(nn.Module):

    def __init__(self, hpo_graph, dise_hpo_freq, propagated_freq):
        super().__init__()
        print('Initializing main model...')
        self.all_diseases = list(dise_hpo_freq.keys())
        self.dise_to_idx = {dise: i for i, dise in enumerate(self.all_diseases)}
        self.idx_to_dise = {v: k for k, v in self.dise_to_idx.items()}
        self.num_diseases = len(self.all_diseases)
        self.omim_orpha_to_idx = {
            sub_key: v
            for k, v in self.dise_to_idx.items()
            for sub_key in k.split(';')
        }

        self.hpo_ids = load_text_as_list("data/hpo_ids.txt")
        self.hpo_to_idx = {hpo: i for i, hpo in enumerate(self.hpo_ids)}
        self.idx_to_hpo = {v: k for k, v in self.hpo_to_idx.items()}
        self.num_hpo = len(self.hpo_ids)

        self.hpo_ic_score = self._init_hpo_ic(propagated_freq)

        anc_tensor = torch.from_numpy(self._build_hpo_anc_mtx(hpo_graph)).float()
        self.register_buffer('anc_matrix', anc_tensor)
        self.hpo_nav = HPONavigator(hpo_graph, self.hpo_to_idx)
        self.specificity_prior = ResidualNetworkClassifier(lr_prior_matrix=self._init_specificity_mtx(), residual_for=LR_PRIOR_TYPE)

    def forward(self, mixed_x_1, mixed_x_2):
        disease_logits = self.specificity_prior(mixed_x_1, mixed_x_2)
        return disease_logits

    def test(self, hpo_ids, mask):
        multi_hot_hpos = self.multi_hot_hpos(hpo_ids, mask)
        multi_hot_hpo_with_anc = self.propagate_vector(multi_hot_hpos)
        disease_logits = self.specificity_prior(multi_hot_hpos, multi_hot_hpo_with_anc)
        return disease_logits

    def predict_with_lr(self, hpo_ids, mask):
        multi_hot_hpos = self.multi_hot_hpos(hpo_ids, mask)
        disease_logits = self.specificity_prior.lr_predict(multi_hot_hpos)
        return disease_logits

    def multi_hot_hpos(self, hpo_ids, mask):
        batch_size = hpo_ids.size(0)
        batch_indices = torch.nonzero(mask, as_tuple=True)[0]  # [total_valid]
        valid_hpo_ids = hpo_ids[mask.bool()]  # [total_valid]
        multi_hot_hpos = torch.zeros(batch_size, self.num_hpo)
        multi_hot_hpos[batch_indices, valid_hpo_ids] = 1.0
        return multi_hot_hpos

    def propagate_vector(self, base_multi_hot, anc_val=1.0):
        anc_activations = torch.matmul(base_multi_hot, self.anc_matrix)
        weighted_hpos = (anc_activations > 0).float() * anc_val
        final_vec = torch.max(weighted_hpos, base_multi_hot)
        return final_vec

    def _init_specificity_mtx(self):
        with torch.no_grad():
            lr_matrix = torch.zeros((self.num_diseases, self.num_hpo))
            pre_cal_lr = load_disease_hpo_freq(LR_PRIOR_PATH)
            for dise, freqs in tqdm(pre_cal_lr.items(), 'Loading Pre-calculated LR'):
                if dise not in self.dise_to_idx:
                    continue
                dise_idx = self.dise_to_idx[dise]
                valid_hpos = [(self.hpo_to_idx[hpo], v) for hpo, v in freqs.items() if hpo in self.hpo_to_idx]
                if valid_hpos:
                    hpo_indices = [hpo_idx for hpo_idx, _ in valid_hpos]
                    values = [v for _, v in valid_hpos]
                    lr_matrix[dise_idx, hpo_indices] = torch.tensor(values)
            return lr_matrix.log()

    def _build_hpo_anc_mtx(self, hpo_graph):
        anc_mtx = np.zeros((self.num_hpo, self.num_hpo), dtype=bool)
        valid_hpo_set = set(self.hpo_ids)
        for i, term in tqdm(enumerate(self.hpo_ids), 'Building HPO Ancestor Matrix'):
            if term not in hpo_graph:
                anc_mtx[i, i] = True
                continue
            ancestors = nx.descendants(hpo_graph, term) | {term}
            valid_ancestors = ancestors.intersection(valid_hpo_set)
            for anc in valid_ancestors:
                j = self.hpo_to_idx[anc]
                anc_mtx[i, j] = True
        return anc_mtx

    def _init_hpo_ic(self, propagated_freq):
        hpo_sum_freq = torch.zeros(self.num_hpo, device=device)
        for disease_id, hpo_dict in tqdm(propagated_freq.items(), 'Computing HPO IC'):
            for hpo_id, freq in hpo_dict.items():
                if hpo_id in self.hpo_to_idx:
                    idx = self.hpo_to_idx[hpo_id]
                    hpo_sum_freq[idx] += freq
        eps = 1e-8
        probs = (hpo_sum_freq + eps) / (self.num_diseases + eps)
        ic_scores = -torch.log(probs)
        return ic_scores


class HPONavigator:
    def __init__(self, hpo_graph, hpo_to_idx):
        print("Initializing HPONavigator...")
        self.graph = hpo_graph
        self.hpo_to_idx = hpo_to_idx
        self.idx_to_hpo = {v: k for k, v in hpo_to_idx.items()}
        self.parent_dict = {k: list(self.graph.successors(k)) for k, v in hpo_to_idx.items()}
        self.child_dict = {k: list(self.graph.predecessors(k)) for k, v in hpo_to_idx.items()}

    def get_parent(self, hpo_id):
        return self.parent_dict.get(hpo_id, [])

    def get_child(self, hpo_id):
        return self.child_dict.get(hpo_id, [])

    def get_random_parent(self, hpo_id):
        parents = self.parent_dict.get(hpo_id, [])
        return random.choice(parents) if parents else None

    def get_random_child(self, hpo_id):
        children = self.child_dict.get(hpo_id, [])
        return random.choice(children) if children else None


class DynamicPatientSimulator:
    def __init__(self, model, kde_model, dise_hpo_freq):
        self.model = model
        self.kde_model = kde_model
        self.dise_hpo_freq = dise_hpo_freq
        self.num_hpo = model.num_hpo

    def generate_epoch_data(self, num_per_disease=10, batch_size=1024):
        self.model.eval()
        num_diseases = self.model.num_diseases
        total_patients = num_diseases * num_per_disease

        dist_samples = self.kde_model.sample(total_patients)
        dist_samples = np.round(dist_samples).astype(int)
        dist_samples = np.clip(dist_samples, 0, None)

        base_vecs = []
        labels = []

        sample_idx = 0
        for dise_id, dise_idx in tqdm(self.model.dise_to_idx.items(), 'Generating Simulated Patients (Simplified)'):
            anno_hpos = list(self.dise_hpo_freq.get(dise_id, {}).keys())
            if not anno_hpos:
                continue

            for _ in range(num_per_disease):
                dist = dist_samples[sample_idx % len(dist_samples)]
                sample_idx += 1
                n_accurate = max(1, dist[0])
                n_noise = dist[1] if len(dist) > 1 else 0
                selected_hpos = random.sample(anno_hpos, min(n_accurate, len(anno_hpos)))
                if n_noise > 0:
                    noise_hpos = random.sample(self.model.hpo_ids, min(n_noise, len(self.model.hpo_ids)))
                    selected_hpos.extend(noise_hpos)
                vec = torch.zeros(self.num_hpo)
                indices = [self.model.hpo_to_idx[h] for h in selected_hpos if h in self.model.hpo_to_idx]
                vec[indices] = 1.0
                base_vecs.append(vec)
                labels.append(dise_idx)

        base_labels = torch.tensor(labels)
        total_patients = len(base_vecs)

        final_base_vecs = []
        final_anc_vecs = []
        final_soft_labels = []

        with torch.no_grad():
            for i in tqdm(range(0, total_patients, batch_size), 'Propagating and Formatting'):
                b_bases = torch.stack(base_vecs[i:i + batch_size]).to(device)
                b_base_anc = self.model.propagate_vector(b_bases)

                y_soft = torch.zeros(b_bases.size(0), num_diseases, device=device)
                batch_labels = base_labels[i:i + batch_size].to(device)
                y_soft.scatter_(1, batch_labels.unsqueeze(1), 1.0)

                final_base_vecs.append(b_bases.to_sparse().cpu())
                final_anc_vecs.append(b_base_anc.to_sparse().cpu())
                final_soft_labels.append(y_soft.cpu())

                del b_bases, b_base_anc, y_soft

        del base_vecs
        return torch.cat(final_base_vecs, dim=0), torch.cat(final_anc_vecs, dim=0), torch.cat(final_soft_labels, dim=0)
