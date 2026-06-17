import json
import re
import time
import requests

from config import OPENAI_API_BASE, OPENAI_API_KEY, GPT_MODEL, RECOMMEND_TOP_N, PHENOTYPE_HPOA_PATH

TOP_K = 5
TOP_K_JSON_CONTENT = ',\n'.join([f'"Rank {i+1}": "[Disease Code]-[Disease Name]"' for i in range(TOP_K)])

SYS_MSG = f"""Role:
You are an expert Medical Diagnostic Consultant specializing in rare diseases (RDs). 
Your task is to perform a high-precision differential diagnosis by synthesizing multi-source computational model outputs with patient-specific phenotypic evidence.

Context & Task:
You will be provided with a patient's clinical profile (primarily HPO terms).
Four specialized diagnostic models have analyzed this case:
1. Deep Learning (DL): Pattern recognition across vast disease-phenotype datasets.
2. Likelihood Ratio (LR): Statistical significance scores calculated from the Knowledge Base.
3. Zero-shot LLM Diagnosis: Heuristic reasoning based on broad medical literature.
4. Patient Bank Retrieval (PBR): Case-based matching against historical patient records.
The results of these models are integrated into a "Robust Rank Aggregation (RRA) List". Your goal is to critically evaluate this list and select the Top {TOP_K} most plausible diagnoses.

Understanding Model Paradigms:
- DL & LR (Reliable Generalists): Systematic models with robust coverage. They serve as your primary diagnostic baseline.
- PBR (Case-Based Specialists): "Evidence-by-precedent." Its scope is limited to known historical cases, but when a match is found, its precision is unparalleled. High PBR evidence is a dominant diagnostic signal.
- LLM Zero-shot (Creative Reasoning): Provides "outside-the-box" insights. It is valuable for identifying rare manifestations or latent connections that rigid statistical models might overlook.

Key Experimental Insight:
- Union Recall Advantage: Internal benchmarks show that while the RRA Fused List has significantly higher Recall@N than any single model, the Union Recall@N across all four single models is significantly higher than RRA list. 
- Complementarity of Single Paradigm: DL and LR paradigms are highly complementary to PBR. DL and LR paradigms also show significant synergy with Zero-shot LLM Diagnosis.
- Database Constraints: DL and LR cover the full RD spectrum; while PBR can only recommend diagnoses that already exist in its patient database.

Your Objective:
Analyze the clinical evidence to select the Top {TOP_K} most likely diseases from the RRA Fused List. 
You can use the provided evidence to justify whether the RRA ranking should be maintained or if a disease ranked lower in the RRA list (but supported by other models or case evidence) is more plausible.

Clinical Reasoning Guidelines:
- Phenotype Matching: Prioritize diseases where the "Directly Matched Phenotypes" cover the most specific or pathognomonic symptoms of the patient.
- Knowledge Base Consistency: Weigh "Always Present (100%)" matches much more heavily than "Common" ones.
- PBR Asymmetry: Many valid candidate diseases may not have corresponding historical patients in the PBR library. The absence of PBR evidence should not be interpreted as evidence against a diagnosis.
- Incomplete Query: Notice that patient phenotypic query usually is incomplete. The absence of "Always Present" terms should not be treated as a definitive disqualifier.

Output Requirements:
1. Chain of Thought (CoT): Provide a step-by-step clinical reasoning for your selection. The superiority of the RRA fused result and the complementarity of individual diagnostic paradigms should be fully considered.
2. Final Selection: At the very end of your output, first write the exact plain text phrase "Final Selection:" on a new line, followed by the final ranking list of Top {TOP_K} diseases in the following JSON format:
```json
{{
{TOP_K_JSON_CONTENT}
}}
"""


class Reflection:
    def __init__(self, hpo_graph):
        self.hpo_id_to_name = {id_: data.get('name') for id_, data in hpo_graph.nodes(data=True)}

    def run_llm_reflection_demo(self, case_info, model_preds):
        patient_hpos = set([self.hpo_id_to_name.get(i, i) for i in case_info['term_id']])
        patient_demography = case_info.get('demography', None)

        dl_topn = model_preds.get('DL', [])[:RECOMMEND_TOP_N]
        lr_topn = model_preds.get('LR', [])[:RECOMMEND_TOP_N]
        pbr_topn = model_preds.get('PBR', [])[:RECOMMEND_TOP_N]
        llm_topn = model_preds.get('LLM', [])[:RECOMMEND_TOP_N]
        rra_topn = model_preds.get('RRA', [])[:RECOMMEND_TOP_N]

        diagnosis_detail_string = "[Disease-related Phenotypic Information Simplified Placeholder]"

        msg = f"""
### Patient Demography (sex: M-Male, F-Female. age: Y-Year, M-Month, D-Day, H-Hour)
- {patient_demography}\n
### Query Phenotypes
- {patient_hpos}\n
### Diagnostic Predictions from Four Different Models
**1. Deep Learning:**
{dl_topn}\n
**2. Likelihood Ratio:**
{lr_topn}\n
**3. LLM Zero-shot Diagnosis:**
{llm_topn}\n
**4. PBR (Patient-Based Retrieval) with Detailed Evidence:**
*This method identifies similar historical cases to support the diagnosis.*
{pbr_topn}\n
### The Robust Rank Aggregation Results of Above Four Models:
{rra_topn}\n
### Disease-related Phenotypic Information (Detailed Analysis) with Disease Code:
*Order based on Robust Rank Aggregation results.*
{diagnosis_detail_string}\n

Let's reason the Top {TOP_K} most likely diseases and present them at the vary end of your response in the agreed-upon JSON format:"""

        llm_ans = llm_query(msg)
        llm_top_k = self.extract_json_results(llm_ans)
        rest_rra_topn = [i for i in rra_topn if i not in llm_top_k]
        combined_top_n = (llm_top_k + rest_rra_topn)[:RECOMMEND_TOP_N]

        return combined_top_n, llm_ans

    def extract_json_results(self, model_output):
        if model_output is None:
            return []
        try:
            json_match = re.search(r'(\{.*?\})', model_output, re.DOTALL | re.MULTILINE)
            if json_match:
                data = json.loads(json_match.group(1))
                results = [data.get(f"Rank {i+1}") for i in range(TOP_K)]
                return [res for res in results if res is not None]
        except Exception:
            pass
        return []


def llm_query(usr_msg):
    gpt_url = "{}/chat/completions".format(OPENAI_API_BASE)
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer {}".format(OPENAI_API_KEY)
    }
    data = {
        "model": GPT_MODEL,
        "messages": [
            {"role": "system", "content": SYS_MSG},
            {"role": "user", "content": usr_msg}
        ],
        "stream": False
    }
    for retry in range(3):
        try:
            response = json.loads(requests.post(gpt_url, headers=headers, json=data).text)
            return response['choices'][0]['message']['content']
        except Exception as e:
            time.sleep(1)
            continue
    return None
