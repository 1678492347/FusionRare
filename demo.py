import time
from concurrent.futures import ThreadPoolExecutor
import obonet
import torch

from llm_diag import llm_top_10
from llm_reflection import Reflection
from model.main import DiagDL
from patient_base import PatientLibrary
from rra import RobustRankAggreg
from utils.utils import *
from config import device, RECOMMEND_TOP_N, ANNOTATION_FILE_PATH, HP_OBO_PATH, RNC_CHECKPOINT

torch.set_default_device(device)


def run_single_case_demo(model, case, disease_vocab_inv, top_k=200):
    model.eval()
    hpo_indices = [hpo_vocab[h] for h in case['term_id'] if h in hpo_vocab]
    hpo_tensor = torch.tensor([hpo_indices], device=device)
    mask = torch.ones_like(hpo_tensor, device=device)

    with torch.no_grad():
        logits_primary = model.test(hpo_tensor, mask)
        logits_lr = model.predict_with_lr(hpo_tensor, mask)
        full_query_vec = model.multi_hot_hpos(hpo_tensor, mask)
        pbr_results = patient_bank.batch_query_pro(full_query_vec, top_k)

    def get_top_k_names(logits, k):
        top_probs, top_indices = torch.topk(logits, k)
        return [(disease_vocab_inv[idx.item()], top_probs[0][i].item()) for i, idx in enumerate(top_indices[0])]

    preds_dl = get_top_k_names(logits_primary, top_k)
    preds_lr = get_top_k_names(logits_lr, top_k)

    return preds_dl, preds_lr, [(res['disease'], res['score']) for res in pbr_results]


if __name__ == '__main__':
    """
    #############################
    Load the knowledge base and patient data, and initialize models.
    #############################
    """
    hpo_graph = obonet.read_obo(HP_OBO_PATH)
    dise_hpo_freq = load_disease_hpo_freq(ANNOTATION_FILE_PATH)
    hpo_id_to_name = {id_: data.get('name') for id_, data in hpo_graph.nodes(data=True)}

    """Main Model"""
    diag_dl = DiagDL(hpo_graph, dise_hpo_freq)
    state_dict = torch.load(f"{RNC_CHECKPOINT}", map_location=device)
    state_dict_fp32 = {
        k: v.float() if v.is_floating_point() else v
        for k, v in state_dict.items()
    }
    diag_dl.load_state_dict(state_dict_fp32, strict=False)
    diag_dl.to(device)
    hpo_vocab = diag_dl.hpo_to_idx
    """Patient Retriever Module"""
    case_of_pb = load_json('data/RPB_sample.json')  # The complete RPB can be downloaded manually from data sources.
    patient_bank = PatientLibrary(diag_dl, case_of_pb)
    """Reflection Module"""
    llm_reflection = Reflection(hpo_graph)
    """
    #############################
    case test
    #############################
    """
    start = time.time()
    demo_case = {
        'patient_id': 'DEMO-001',
        'term_id': ['HP:0007843', 'HP:0001513', 'HP:0000608', 'HP:0000486', 'HP:0001328', 'HP:0000510', 'HP:0001263'],
        'demography': None,
        'diagnosis': ['OMIM:209900'],
    }

    """L1 - Source Diagnosis"""
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_demo = executor.submit(
            run_single_case_demo,
            diag_dl,
            demo_case,
            diag_dl.idx_to_dise,
            top_k=RECOMMEND_TOP_N
        )
        future_llm = executor.submit(
            llm_top_10,
            demo_case['term_id'],
            hpo_id_to_name
        )
        dl_rank, lr_rank, pbr_rank = future_demo.result()
        llm_rank = future_llm.result()
        llm_rank = [(k, v) for k, v in zip(llm_rank, [1.0 - (pos * 0.5 / (len(llm_rank) - 1)) for pos in range(len(llm_rank))])]

    L1 = [dl_rank, lr_rank, pbr_rank, llm_rank]
    L1_name_list = [[j[0] for j in i] for i in L1]
    L1_score_list = [[j[1] for j in i] for i in L1]

    """L2 - Statistical Aggr"""
    L2_base_aggr = ["min", "mean", "geom.mean", "combmnz"]
    L2_base_res = {}
    for model_name in L2_base_aggr:
        if model_name == "combmnz":
            L2_base_res[model_name] = RobustRankAggreg.aggregate_ranks(L1_name_list, L1_score_list, method=model_name)['Name'].values.tolist()[:RECOMMEND_TOP_N]
        else:
            L2_base_res[model_name] = RobustRankAggreg.aggregate_ranks(L1_name_list, method=model_name, full=False)['Name'].values.tolist()[:RECOMMEND_TOP_N]
    L2_base_aggr_weight = {
        'geom.mean': 0.5,
        'mean': 0.5,
        'combmnz': 0.5,
        'min': 1.0
    }
    L2_base_res_rank_mat = RobustRankAggreg.rank_matrix([L2_base_res[aggr] for aggr in L2_base_aggr])
    L2 = RobustRankAggreg.aggregate_weighted_geom_mean(L2_base_res_rank_mat, [L2_base_aggr_weight[aggr] for aggr in L2_base_aggr])
    L2_top_n_name = L2['Name'].values.tolist()[:RECOMMEND_TOP_N]
    L2_top_n_score = L2['Score'].values.tolist()[:RECOMMEND_TOP_N]

    """L3 - Agentic Reflection"""
    L3, llm_ans = llm_reflection.run_llm_reflection_demo(
        case_info=demo_case,
        model_preds={
            'DL': L1_name_list[0],
            'LR': L1_name_list[1],
            'PBR': L1_name_list[2],
            'LLM': L1_name_list[3],
            'RRA': L2_top_n_name
        }
    )

    print(time.time()-start)
    print()
