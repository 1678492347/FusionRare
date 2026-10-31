import math
import time
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import obonet
import torch

from llm_diag import llm_top_10
from llm_reflection import Reflection
from model.main import DiagDL
from patient_base import PatientLibrary
from rra import RobustRankAggreg
from utils.utils import *
from config import device, RECOMMEND_TOP_N, ANNOTATION_FILE_PATH, HP_OBO_PATH, RNC_CHECKPOINT, RPB_SAMPLE, \
    PROPAGATED_FREQ_PATH

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
        pbr_results = patient_bank.batch_query_pro(full_query_vec, top_k, get_patient_info=True)[0]

    def get_top_k_names(logits, k):
        top_probs, top_indices = torch.topk(logits, k)
        return [(disease_vocab_inv[idx.item()], top_probs[0][i].item()) for i, idx in enumerate(top_indices[0])]

    preds_dl = get_top_k_names(logits_primary, top_k)
    preds_lr = get_top_k_names(logits_lr, top_k)

    return preds_dl, preds_lr, [(res['disease'], res['score']) for res in pbr_results], pbr_results


def mapping_diag_to_dise(dataset, model):
    return [{
        **i,
        'diagnosis': model.idx_to_dise.get(model.omim_orpha_to_idx.get(i['diagnosis']))
    } for i in dataset]


def parse_missed_valid_mapping():
    try:
        with open('data/missed_valid_mapping.json', 'r') as f:
            data = json.load(f)
        data = [tuple(item) for item in data]
        G = nx.Graph()
        G.add_edges_from(data)
        G.remove_nodes_from([node for node in G.nodes() if node not in disease_vocab])
        data = [list(sorted(component)) for component in nx.connected_components(G)]
        return {j: ';'.join(i) for i in data for j in i}
    except:
        return {}


def dedup(rank_list):
    rank_list = [i if i[0] not in missed_valid_mapping else (missed_valid_mapping[i[0]], i[1]) for i in rank_list]
    return list({item[0]: item for item in rank_list}.values())


def merge_genotype_LR(row, score_dict):
    names = row['Name'].split(';')
    scores = list(filter(None, [score_dict.get(name, None) for name in names]))
    max_score = max(scores) if scores else 0
    return row['Score'] + max(math.log10(max_score), 0) if max_score > 0 else row['Score']


def get_genotype_exp(row, score_dict, exp_dict):
    try:
        names = row['Name'].split(';')
        scores = [score_dict.get(name, None) for name in names]
        valid_pairs = [(name, score) for name, score in zip(names, scores) if score is not None]
        best_name, _ = max(valid_pairs, key=lambda x: x[1])
        return exp_dict[best_name].replace('<sub>', '').replace('</sub>', '')
    except:
        return 'None'


def load_lirical_result(vcf_id):
    """Pseudocode: returns HpoCase instance from LIRICAL's phenotype-genotype joint analysis."""
    with open(f"data/{vcf_id}.json", 'r', encoding='utf-8') as f:
        res = json.load(f)
    return res


if __name__ == '__main__':
    """
    #############################
    Load the knowledge base and patient data, and initialize models.
    #############################
    """
    hpo_graph = obonet.read_obo(HP_OBO_PATH)
    dise_hpo_freq = load_disease_hpo_freq(ANNOTATION_FILE_PATH)
    propagated_freq = load_disease_hpo_freq(PROPAGATED_FREQ_PATH)
    hpo_id_to_name = {id_: data.get('name') for id_, data in hpo_graph.nodes(data=True)}
    missed_valid_mapping = parse_missed_valid_mapping()

    """Main Model"""
    diag_dl = DiagDL(hpo_graph, dise_hpo_freq, propagated_freq)
    state_dict = torch.load(RNC_CHECKPOINT, map_location=device)
    state_dict_fp32 = {
        k: v.float() if v.is_floating_point() else v
        for k, v in state_dict.items()
    }
    diag_dl.load_state_dict(state_dict_fp32, strict=False)
    diag_dl.to(device)
    hpo_vocab = diag_dl.hpo_to_idx
    disease_vocab = diag_dl.omim_orpha_to_idx

    """Patient Retriever Module"""
    case_of_pb = load_json(RPB_SAMPLE)  # The complete RPB can be downloaded manually from data sources.
    patient_bank = PatientLibrary(diag_dl, mapping_diag_to_dise(case_of_pb, diag_dl))

    """Reflection Module"""
    llm_reflection = Reflection(hpo_graph, dise_hpo_freq)

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
        'vcf_id': '23559858-Ajmal-2013-BBS1-IV-5_family_A'
    }

    """L1 - Source Diagnosis"""
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_demo = executor.submit(
            run_single_case_demo,
            diag_dl,
            demo_case,
            diag_dl.idx_to_dise,
        )
        future_llm = executor.submit(
            llm_top_10,
            demo_case['term_id'],
            hpo_id_to_name
        )
        dl_rank, lr_rank, pbr_rank, pbr_full_res = future_demo.result()
        llm_rank = future_llm.result()

    llm_rank = [(k, v) for k, v in zip(llm_rank, [1.0 - (pos * 0.5 / (len(llm_rank) - 1)) for pos in range(len(llm_rank))])]

    L1 = [dedup(dl_rank), dedup(lr_rank), dedup(pbr_rank), dedup(llm_rank)]
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

    VCF_UPLOADED = False
    if VCF_UPLOADED:
        L2['Pheno Rank'] = list(range(1, len(L2) + 1))
        lirical_res = load_lirical_result(demo_case['vcf_id'])
        lirical_disease_genotype_LR = {i['diseaseDatabaseId']['value']: i['genotypeLR'] for i in lirical_res['results']}
        lirical_disease_genotype_exp = {i['diseaseDatabaseId']['value']: i['genotypeExplanation'] for i in lirical_res['results']}
        lirical_disease_gene = {i['diseaseDatabaseId']['value']: i['geneSymbol'] for i in lirical_res['results']}

        L2['Merged Score'] = L2.apply(merge_genotype_LR, axis=1, score_dict=lirical_disease_genotype_LR)
        L2 = L2.sort_values('Merged Score', ascending=False).reset_index(drop=True)
        L2 = L2.iloc[:RECOMMEND_TOP_N]

        L2['Gene Explanation'] = L2.apply(get_genotype_exp, axis=1, score_dict=lirical_disease_genotype_LR, exp_dict=lirical_disease_genotype_exp)
        L2['Gene logLR'] = L2['Merged Score'] - L2['Score']

        L2_top_n_name = L2['Name'].values.tolist()
        L2_top_n_score = L2['Score'].values.tolist()
        rra_detail = L2.loc[:, ['Name', 'Score', 'Gene logLR', 'Merged Score', 'Gene Explanation', 'Pheno Rank']].to_dict(orient='records')
    else:
        L2_top_n_name = L2['Name'].values.tolist()[:RECOMMEND_TOP_N]
        L2_top_n_score = L2['Score'].values.tolist()[:RECOMMEND_TOP_N]
        rra_detail = None

    """L3 - Agentic Reflection"""
    L3, llm_ans = llm_reflection.run_llm_reflection_demo(
        case_info=demo_case,
        model_preds={
            'DL': L1_name_list[0],
            'LR': L1_name_list[1],
            'PBR': L1_name_list[2],
            'LLM': L1_name_list[3],
            'RRA': L2_top_n_name
        },
        patient_bank_rank=pbr_full_res,
        missed_valid_mapping=missed_valid_mapping,
        rra_detail=rra_detail
    )
    print(L3)
