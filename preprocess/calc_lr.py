import json
from tqdm import tqdm

from config import ANNOTATION_FILE_PATH, LR_PRIOR_PATH


def calculate_annotation_overall_prob(disease_data):
    N = len(disease_data)
    if N == 0:
        return {}
    hpo_sum_freq = {}
    for annotations in disease_data.values():
        for hpo_id, freq in annotations.items():
            hpo_sum_freq[hpo_id] = hpo_sum_freq.get(hpo_id, 0.0) + freq
    return {hpo_id: sum_freq / N for hpo_id, sum_freq in hpo_sum_freq.items()}


def calculate_annotation_lr(disease_data, hpo_overall_prob):
    DEFAULT_BACKGROUND_PROBABILITY = 1.0 / 10000
    disease_2_hpo_lr = {}

    for disease_id, annotations in tqdm(disease_data.items(), 'Calculating Annotation-level LR'):
        hpo_lr = {}
        for hpo_id, freq in annotations.items():
            background_prob = hpo_overall_prob.get(hpo_id, DEFAULT_BACKGROUND_PROBABILITY)
            lr = freq / max(background_prob, 1e-8)
            hpo_lr[hpo_id] = lr
        disease_2_hpo_lr[disease_id] = hpo_lr

    return disease_2_hpo_lr


def process_all_diseases(json_path, output_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        disease_data = json.load(f)
    hpo_overall_prob = calculate_annotation_overall_prob(disease_data)
    disease_2_hpo_lr = calculate_annotation_lr(disease_data, hpo_overall_prob)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(disease_2_hpo_lr, f, indent=2)


if __name__ == "__main__":
    process_all_diseases(ANNOTATION_FILE_PATH, LR_PRIOR_PATH)