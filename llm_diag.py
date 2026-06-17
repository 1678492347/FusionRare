import json
import re
import time

import numpy as np
import requests

from config import OPENAI_API_BASE, OPENAI_API_KEY, GPT_MODEL
from disease_retriever.retriever import embedded_disease_faiss_index, embedding_model, embedded_disease_documents


matched_map = {}


def parse_llm_answer(text):
    if '1.' in text:
        match = re.search(r'\d+\.\s', text)
    elif '1)' in text:
        match = re.search(r'\d+\)\s', text)
    else:
        return None
    if match:
        text = text[match.start():]
        if '1.' in text:
            re1 = re.split(r'\d+\.\s', text)[1:]
        else:
            re1 = re.split(r'\d+\)\s', text)[1:]
        return list(filter(None, [segment.split('\n')[0].strip() if '\n' in segment else segment.strip() for segment in re1]))
    return None


def llm_top_10(phe_ids, hpo_id_to_name):
    gpt_url = "{}/chat/completions".format(OPENAI_API_BASE)
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer {}".format(OPENAI_API_KEY)
    }
    question = 'Please give a ranked list with 10 differential diagnosis names (rare disease names only) based on the HPOs provided below:\n{}\n(Give a numerical list of disease names in plain text, without disease descriptions and without other special formatting such as bold)'.format(
        '\n'.join(['{} - {}'.format(phe_id, hpo_id_to_name.get(phe_id)) for phe_id in phe_ids])
    )
    data = {
        "model": GPT_MODEL,
        "messages": [{"role": "user", "content": question}],
        "stream": False
    }
    names = []
    for retry in range(3):
        try:
            response = json.loads(requests.post(gpt_url, headers=headers, json=data).text)
            llm_ans = response['choices'][0]['message']['content']
        except Exception as e:
            time.sleep(1)
            continue
        names = parse_llm_answer(llm_ans)
        if not names: continue
        break

    for name in names:
        if name.lower() in matched_map: continue
        similarities, indices = embedded_disease_faiss_index.search(np.array(list(embedding_model.encode([name.lower()]))[0]).astype(np.float32).reshape(1, -1), 10)
        unique_metadata_list = [embedded_disease_documents[idx]['unique_metadata'] for idx in indices[0]]
        candidate_name = unique_metadata_list[0]['text']
        candidate_id = unique_metadata_list[0]['disease_id']
        record = {
            'gpt_name': name,
            'matched': True,
            'disease_name': candidate_name,
            'disease_id': candidate_id,
            'similarity': similarities[0][0]
        }
        matched_map[name.lower()] = record
    llm_top10 = []
    for name in names:
        llm_top10.append(matched_map[name.lower()]['disease_id'])

    return list(dict.fromkeys(llm_top10))


