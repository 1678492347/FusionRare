import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from config import *


def load_and_flatten_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        disease_map = json.load(f)

    flattened_data = []
    for disease_id, synonyms in disease_map.items():
        for synonym in synonyms:
            flattened_data.append((disease_id, synonym.strip()))
    return flattened_data


def create_vector_database(data, output_file):

    model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=HF_CACHE_FOLDER)

    embedded_documents = []

    for i in tqdm(range(0, len(data), BATCH_SIZE), desc="Processing"):
        batch = data[i: i + BATCH_SIZE]
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]

        embeddings = model.encode(texts, show_progress_bar=False)

        for j in range(len(batch)):
            entry = {
                'embedding': embeddings[j].astype(np.float32),
                'unique_metadata': {
                    'disease_id': ids[j],
                    'text': texts[j]
                }
            }
            embedded_documents.append(entry)

    np.save(output_file, embedded_documents, allow_pickle=True)
    print(f"Saved at: {output_file}")
    print(f"Total: {len(embedded_documents)}")


if __name__ == "__main__":
    BATCH_SIZE = 128
    processed_data = load_and_flatten_data(DISEASE_NAME_PATH)
    if processed_data:
        create_vector_database(processed_data, DISEASE_NAME_EMB_PATH)
    else:
        print("Error!")
