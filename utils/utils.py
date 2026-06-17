import json
import os
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

import config


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_json(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} don't exist")
    with open(file_path) as f:
        data = json.load(f)
    return data


def read_json_lines(file_path):
    json_objects = []
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            for line in file:
                line = line.strip()
                if line:
                    try:
                        json_obj = json.loads(line)
                        json_objects.append(json_obj)
                    except json.JSONDecodeError as e:
                        print("JSONDecodeError")
                        continue
    except FileNotFoundError:
        print(f"FileNotFoundError")
    except Exception as e:
        print(f"Exception: {e}")
    return json_objects


def unique_patient(data):
    unique_dict = {}
    for item in data:
        key = (item['diagnosis'], tuple(sorted(item['term_id'])))
        unique_dict[key] = item
    return list(unique_dict.values())


def load_text_as_list(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} don't exist")
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]


def load_disease_hpo_freq(file_path, dataset=config.DATASET):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} don't exist")
    with open(file_path) as f:
        disease_hpo_freq = json.load(f)
    if dataset == ["ORPHA"]:
        filtered_data = {
            key: value for key, value in disease_hpo_freq.items()
            if dataset[0] in key
        }
    elif dataset == ["OMIM"]:
        filtered_data = {
            key: value for key, value in disease_hpo_freq.items()
            if dataset[0] in key
        }
    else:
        filtered_data = disease_hpo_freq
    return dict(sorted(filtered_data.items(), key=lambda x: x[0]))


def get_data_path(filename):
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(utils_dir)
    return os.path.join(base_dir, "data", filename)


def parse_pheno_quality(patients, annotated_freq, hpo_graph):
    # Simplified patient phenotype quality analysis for demonstration.
    total_count_list = []
    anno_count_list = []
    noise_count_list = []
    dise_anno_count_list = []
    for record in tqdm(patients, 'Parsing Patient HPO Quality'):
        try:
            target = record['diagnosis']
            phe = record['term_id']
            target_anno_phe = set(annotated_freq.get(target).keys())
            anno_count = 0
            noise_count = 0
            for p in phe:
                if p in target_anno_phe:
                    anno_count += 1
                else:
                    noise_count += 1
            total_count_list.append(len(phe))
            anno_count_list.append(anno_count)
            noise_count_list.append(noise_count)
            dise_anno_count_list.append(len(target_anno_phe))
        except Exception:
            continue
    return [total_count_list, dise_anno_count_list, anno_count_list, noise_count_list]
