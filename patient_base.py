import torch
import numpy as np


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

    def batch_query_pro(self, query_vecs, top_k=10, get_patient_info=False):
        q_anc_vecs = self.model.propagate_vector(query_vecs)  # [Batch, Num_HPO]

        weighted_patient_anc = self.patient_anc_mtx * self.ic_vector  # [Lib, Num_HPO]
        sim_q_hk = torch.matmul(query_vecs, weighted_patient_anc.T)

        weighted_q_anc = q_anc_vecs * self.ic_vector  # [Batch, Num_HPO]
        sim_hk_q = torch.matmul(weighted_q_anc, self.patient_hpo_mtx.T)

        score_matrix = 0.5 * (sim_q_hk + sim_hk_q)

        batch_size = score_matrix.size(0)
        disease_scores = torch.full(
            (batch_size, self.num_diseases_in_lib),
            -1e9,
            device=self.device
        )

        disease_indices_expanded = self.patient_to_dise_idx.expand(batch_size, -1)
        disease_scores.scatter_reduce_(
            dim=1,
            index=disease_indices_expanded,
            src=score_matrix,
            reduce='amax',
            include_self=False
        )
        pat_to_dis_cpu = self.patient_to_dise_idx.cpu()
        is_best_patient_cpu = None
        if get_patient_info:
            max_scores_per_patient_pos = disease_scores.gather(1, disease_indices_expanded)
            is_best_patient_cpu = (score_matrix == max_scores_per_patient_pos).cpu()

        top_scores, top_indices = torch.topk(disease_scores, k=top_k, dim=1)

        top_scores_cpu = top_scores.cpu().numpy()
        top_indices_cpu = top_indices.cpu().numpy()

        batch_results = []
        for i in range(batch_size):
            if not get_patient_info:
                results = [
                    {'disease': self.idx_to_dise[idx], 'score': float(score)}
                    for idx, score in zip(top_indices_cpu[i], top_scores_cpu[i])
                ]
                batch_results.append(results)
            else:
                results = []
                current_query_best_mask = is_best_patient_cpu[i]
                for rank in range(top_k):
                    d_idx = top_indices_cpu[i, rank]
                    score = top_scores_cpu[i, rank]
                    best_pat_indices = torch.where(
                        (pat_to_dis_cpu == d_idx) & current_query_best_mask
                    )[0].tolist()
                    best_pat_hpo_indices = [
                        torch.where(self.patient_hpo_mtx[idx] > 0)[0].tolist()
                        for idx in best_pat_indices
                    ]
                    results.append({
                        "disease": self.idx_to_dise[d_idx],
                        "score": float(score),
                        "best_patient_indices": best_pat_indices,
                        "best_patient_hpos": [
                            [self.model.idx_to_hpo[idx] for idx in sublist]
                            for sublist in best_pat_hpo_indices
                        ]
                    })
                batch_results.append(results)
        return batch_results
