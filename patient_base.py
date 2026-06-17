import torch
import numpy as np
from collections import defaultdict


class PatientLibrary:
    def __init__(self, model, patient_data_list):
        self.model = model
        self.device = next(model.parameters()).device

        processed_patients = patient_data_list

        self.patient_hpo_mtx, self.patient_anc_mtx, self.diagnoses = self._build_matrices(processed_patients)

        self.dise_to_idx = {d: i for i, d in enumerate(sorted(list(set(self.diagnoses))))}
        self.idx_to_dise = {i: d for d, i in self.dise_to_idx.items()}
        self.num_diseases_in_lib = len(self.dise_to_idx)

        self.patient_to_dise_idx = torch.tensor(
            [self.dise_to_idx[d] for d in self.diagnoses],
            device=self.device
        )

        self.ic_vector = self.model.hpo_ic_score.to(self.device)  # [num_hpo]

    def _build_matrices(self, patients):
        num_p = len(patients)
        hpo_mtx = torch.zeros((num_p, self.model.num_hpo), device=self.device)
        diagnoses = []
        for i, p in enumerate(patients):
            indices = [self.model.hpo_to_idx[h] for h in p['term_id'] if h in self.model.hpo_to_idx]
            hpo_mtx[i, indices] = 1.0
            diagnoses.append(p['diagnosis'])

        anc_mtx = self.model.propagate_vector(hpo_mtx)
        return hpo_mtx, anc_mtx, np.array(diagnoses)

    def batch_query_pro(self, query_vecs, top_k=10):
        q_anc_vecs = self.model.propagate_vector(query_vecs)
        weighted_patient_anc = self.patient_anc_mtx * self.ic_vector
        sim_q_hk = torch.matmul(query_vecs, weighted_patient_anc.T)

        weighted_q_anc = q_anc_vecs * self.ic_vector
        sim_hk_q = torch.matmul(weighted_q_anc, self.patient_hpo_mtx.T)

        score_matrix = 0.5 * (sim_q_hk + sim_hk_q)
        score_matrix_cpu = score_matrix.cpu().numpy()

        batch_results = []
        for i in range(score_matrix.size(0)):
            row_scores = score_matrix_cpu[i]
            disease_best_scores = defaultdict(float)

            for score, disease in zip(row_scores, self.diagnoses):
                if score > disease_best_scores[disease]:
                    disease_best_scores[disease] = score

            sorted_ranking = sorted(disease_best_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

            results = []
            for d_name, score in sorted_ranking:
                item = {
                    "disease": d_name,
                    "score": float(score)
                }
                results.append(item)
            batch_results.append(results)

        return batch_results
