import json
import re
import time

import pandas as pd
import requests

from config import OPENAI_API_BASE, OPENAI_API_KEY, GPT_MODEL, RECOMMEND_TOP_N, PHENOTYPE_HPOA_PATH

TOP_K = 5
TOP_K_JSON_CONTENT = ',\n'.join([f'"Rank {i+1}": "[Disease Code]-[Disease Name]"' for i in range(TOP_K)])

SYS_MSG_1 = f"""Role:
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

SYS_MSG_2 = f"""Role:
You are an expert Medical Diagnostic Consultant specializing in rare diseases (RDs) and Genomic Medicine. 
Your task is to perform a high-precision differential diagnosis by synthesizing multi-source computational model outputs (Phenotypic & Genotypic) with patient-specific phenotypic evidence.

Context & Task:
You will be provided with a patient's clinical profile (primarily HPO terms as well as pathogenic genes identified from its sequencing data).
Four specialized diagnostic models have analyzed this case (analyzing only phenotypic information):
1. Deep Learning (DL): Pattern recognition across vast disease-phenotype datasets.
2. Likelihood Ratio (LR): Statistical significance scores calculated from the Knowledge Base.
3. Zero-shot LLM Diagnosis: Heuristic reasoning based on broad medical literature.
4. Patient Bank Retrieval (PBR): Case-based matching against historical patient records.
Another model, known as LIRICAL, is used for genotype LR score analysis, quantifying the pathogenicity of specific variants linked to the disease/gene.
The results of these models (Phenotypic & Genotypic) are integrated into a "Robust Rank Aggregation (RRA) List". Your goal is to critically evaluate this list and select the Top {TOP_K} most plausible diagnoses.

Understanding Model Paradigms:
- DL & LR (Reliable Generalists): Systematic models with robust coverage. They serve as your primary diagnostic baseline.
- PBR (Case-Based Specialists): "Evidence-by-precedent." Its scope is limited to known historical cases, but when a match is found, its precision is unparalleled. High PBR evidence is a dominant diagnostic signal.
- LLM Zero-shot (Creative Reasoning): Provides "outside-the-box" insights. It is valuable for identifying rare manifestations or latent connections that rigid statistical models might overlook.
- LIRICAL (Genotype Integrator): Provides molecular evidence. LogLR > 0 indicates genotypic support. A high genotype LR score (e.g., LogLR > 2.0) constitutes a significant signal of high-confidence pathogenic variant.

Key Experimental Insight:
- Genotype-Phenotype Synergy: LIRICAL\'s genotypic logLR scores has been integrated into the RRA Fused results as strong evidence.
    * The Phenotypic_Score and the Genotypic_Score(logLR) are dimensionally inconsistent; while genotypic logLR>0 indicates genotypic support, phenotypic scores are invariably positive.
    * Disease with genotypic support and phenotypic associations should be prioritized.
    * Disease that exhibit ABSOLUTE no (potential) phenotypic association with the queried patient but has genotypic support should be treated with CAUTION.
- Phenotype-Genotype Diagnosis Advantage: Typically, the RRA fusion list—by integrating both phenotypic and genotypic information—demonstrates Recall@N metrics that are significantly higher than those of any single model.
- Database Constraints: DL and LR cover the full RD spectrum; while PBR can only recommend diagnoses that already exist in its patient database.

Your Objective:
Analyze the clinical evidence to select the Top {TOP_K} most likely diseases from the RRA Fused List. 
You can use the provided evidence to justify whether the RRA ranking should be maintained or if a disease ranked lower in the RRA list (but supported by other models or case evidence) is more plausible.

Clinical Reasoning Guidelines:
- Genotype Primacy: Prioritize diseases where a pathogenic or likely pathogenic variant (high LR) is identified in a causal gene, and the disease demonstrates phenotypic association with the queried phenotypes.
- Phenotype Matching: Prioritize diseases where the "Directly Matched Phenotypes" cover the most specific or pathognomonic symptoms of the patient, while simultaneously not overlooking potential phenotypic associations.
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
    def __init__(self, hpo_graph, dise_hpo_freq):
        self.hpo_id_to_name = {id_: data.get('name') for id_, data in hpo_graph.nodes(data=True)}
        self.dise_hpo_freq = {
            disease: dict(sorted(hpo_dict.items(), key=lambda item: item[1], reverse=True))
            for disease, hpo_dict in dise_hpo_freq.items()
        }   # freq降序排序
        self.orpha_omim2code = {
            sub_key: k
            for k, v in dise_hpo_freq.items()
            for sub_key in k.split(';')
        }
        df = pd.read_csv(PHENOTYPE_HPOA_PATH, sep='\t', comment='#', low_memory=False, header=None)
        df.columns = ['database_id', 'disease_name', 'qualifier', 'hpo_id', 'reference', 'evidence', 'onset',
                      'frequency', 'sex', 'modifier', 'aspect', 'biocuration']
        self.dise_name_map = df.groupby('database_id')['disease_name'].first().to_dict()
        self.dise_code_map = {v: k for k, v in self.dise_name_map.items()}

    def run_llm_reflection_demo(self, case_info, model_preds, patient_bank_rank, missed_valid_mapping, rra_detail=None):
        patient_hpos = set([self.hpo_id_to_name.get(i, i) for i in case_info['term_id']])
        patient_demography = case_info.get('demography', None)

        rl_topn = model_preds['DL'][:RECOMMEND_TOP_N]
        lr_topn = model_preds['LR'][:RECOMMEND_TOP_N]
        pbr_topn = model_preds['PBR'][:RECOMMEND_TOP_N]
        llm_topn = model_preds['LLM'][:RECOMMEND_TOP_N]
        rra_topn = model_preds['RRA'][:RECOMMEND_TOP_N]

        pheno_geno_analysis = False if rra_detail is None else True
        if rra_detail:
            rra_detail = {i['Name']: {
                'geno_exp': i['Gene Explanation'],
                'pheno_score': round(i['Score'], 2),
                'geno_score': round(i['Gene logLR'], 2),
                'comb_score': round(i['Merged Score'], 2),
                'raw_pheno_rank': i['Pheno Rank'],
            } for i in rra_detail}

        PATIENT_PER_DISEASE_LIMIT = 5
        pbr_sim_patients = [i['best_patient_hpos'][:PATIENT_PER_DISEASE_LIMIT] for i in patient_bank_rank[:RECOMMEND_TOP_N]]
        pbr_sim_patients = [[[self.hpo_id_to_name[hpo_id] for hpo_id in pa] for pa in no] for no in pbr_sim_patients]

        pbr_details = []
        pbr_evidence_dict = {}
        PBR_SPECIFIC_LIMIT = 10
        for no, diag_code in enumerate(pbr_topn):
            choosen_code = diag_code.split(';')[-1]
            dise_name = self.dise_name_map[choosen_code]
            evidence_list = []
            max_shared_count = -1
            best_case_no = 1
            best_shared_phenotypes = []
            for sim_no, sim_pa in enumerate(pbr_sim_patients[no]):
                sim_pa = set(sim_pa)
                common = list(sim_pa & patient_hpos)
                specific = list(sim_pa - patient_hpos)
                if len(common) > max_shared_count:
                    max_shared_count = len(common)
                    best_case_no = sim_no + 1
                    best_shared_phenotypes = common[:]
                evidence = (
                    f"      - Case Instance {sim_no + 1}:\n"
                    f"        * Shared with Query: [{', '.join(common) if len(common) > 0 else 'None'}]\n"
                    f"        * Unique to this Case: [{', '.join(specific[:PBR_SPECIFIC_LIMIT])}{f', and so on](Total {len(specific)} unique phenotypes)' if len(specific) > PBR_SPECIFIC_LIMIT else ']'}"
                )
                evidence_list.append(evidence)
            if max_shared_count > 0:
                pbr_evidence_dict[
                    diag_code] = f"- PBR Evidence: Shared {max_shared_count} phenotypes ({', '.join(best_shared_phenotypes)}) with Case Instance {best_case_no} of This Disease."
            else:
                pbr_evidence_dict[
                    diag_code] = "- PBR Evidence: Retrieved cases share no specific phenotypes with the Query."
            item = f"   {no + 1}. {dise_name} [{choosen_code}]\n" + "\n".join(evidence_list)
            pbr_details.append(item)
        pbr_string = "\n".join(pbr_details)

        HPO_LIMIT = 20
        diagnosis_details = []
        for no, diag_code in enumerate(rra_topn):
            all_hpos = list(self.dise_hpo_freq.get(self.orpha_omim2code[diag_code.split(';')[0]], {}).items())
            total_hpo_count = len(all_hpos)
            matched_hpos = [h[0] for h in all_hpos if self.hpo_id_to_name.get(h[0], h[0]) in patient_hpos]
            if matched_hpos:
                matched_str = ", ".join([f"{self.hpo_id_to_name.get(h, h)}" for h in matched_hpos])
            else:
                matched_str = "None"
            visible_hpos = all_hpos[:HPO_LIMIT]
            cats = {
                "Always Present (100%)": [h for h in visible_hpos if h[1] == 1],
                "Very Frequent (80-99%)": [h for h in visible_hpos if 0.8 <= h[1] < 1],
                "Common (30-79%)": [h for h in visible_hpos if 0.3 <= h[1] < 0.8],
                "Uncommon (<30%)": [h for h in visible_hpos if h[1] < 0.3]
            }
            cat_lines = []
            for cat_label, hpo_list in cats.items():
                if hpo_list:
                    hpo_format = ", ".join([f"{self.hpo_id_to_name.get(h[0], h[0])}({h[1]:.2f})" for h in hpo_list])
                    cat_lines.append(f"        * {cat_label}: {hpo_format}")
            etc_suffix = f", and so on. [Total {len(all_hpos)} Related Features]" if total_hpo_count > HPO_LIMIT else ""
            choosen_code = diag_code.split(';')[-1]
            dise_name = self.dise_name_map.get(choosen_code, choosen_code)
            if pheno_geno_analysis:
                detail_block = (
                        f"   {no + 1}. **{dise_name} [{choosen_code}]**\n"
                        f"      - Directly Matched Phenotypes with Query: {matched_str}\n"
                        f"      - Genotype Evidence (LIRICAL Explanation): {rra_detail[diag_code]['geno_exp']}\n"
                        + (f"      {pbr_val}\n" if (pbr_val := pbr_evidence_dict.get(diag_code)) else "")
                        + f"      - Disease Key Features from KB:\n"
                        + "\n".join(cat_details if (cat_details := cat_lines) else ["        * No high-frequency features recorded."])
                        + etc_suffix
                )
            else:
                detail_block = (
                        f"   {no + 1}. **{dise_name} [{choosen_code}]**\n"
                        f"      - Directly Matched Phenotypes with Query: {matched_str}\n"
                        + (f"      {pbr_val}\n" if (pbr_val := pbr_evidence_dict.get(diag_code)) else "")
                        + f"      - Disease Key Features from KB:\n"
                        + "\n".join(cat_details if (cat_details := cat_lines) else ["        * No high-frequency features recorded."])
                        + etc_suffix
                )
            diagnosis_details.append(detail_block)
        diagnosis_detail_string = "\n".join(diagnosis_details)

        if pheno_geno_analysis:
            msg = f"""
### Query Phenotypes
- {'; '.join(patient_hpos)}\n
### Diagnostic Predictions from Five Different Models
**1. Deep Learning:**
{chr(10).join([f"   {no + 1}. {self.dise_name_map[diag_code.split(';')[-1]]} [{diag_code.split(';')[-1]}]" for no, diag_code in enumerate(rl_topn)])}\n
**2. Likelihood Ratio:**
{chr(10).join([f"   {no + 1}. {self.dise_name_map[diag_code.split(';')[-1]]} [{diag_code.split(';')[-1]}]" for no, diag_code in enumerate(lr_topn)])}\n
**3. LLM Zero-shot Diagnosis:**
{chr(10).join([f"   {no + 1}. {self.dise_name_map[diag_code.split(';')[-1]]} [{diag_code.split(';')[-1]}]" for no, diag_code in enumerate(llm_topn)])}\n
**4. PBR (Patient-Based Retrieval) with Detailed Evidence:**
*This method identifies similar historical cases to support the diagnosis.*
{pbr_string}\n
**5. LIRICAL (Genotype-only LR):**
*The pathogenic genotypes and their corresponding log LR scores—identified by LIRICAL based on the input sequencing data—have been integrated into the subsequent RRA results and Disease-related Detailed Analysis.*

### The Robust Rank Aggregation Results of Above Five Models:
*For diagnostic robustness, the logLR scores calculated by LIRICAL are truncated to a minimum value of 0.0, meaning any negative scores are set to 0.*
{chr(10).join([f"   {no + 1}. {self.dise_name_map[diag_code.split(';')[-1]]} (Phenotypic_Score: {rra_detail[diag_code]['pheno_score']}, Genotypic_Score(logLR): {rra_detail[diag_code]['geno_score']}, Phenotype-only Ranking: {rra_detail[diag_code]['raw_pheno_rank']})" for no, diag_code in enumerate(rra_topn)])}\n
### Disease-related Detailed Analysis with Disease Code:
*Order based on Robust Rank Aggregation results.*
{diagnosis_detail_string}\n

Let's reason the Top {TOP_K} most likely diseases and present them at the vary end of your response in the agreed-upon JSON format:"""
        else:
            msg = f"""
### Patient Demography (sex: M-Male, F-Female. age: Y-Year, M-Month, D-Day, H-Hour)
- {patient_demography}\n
### Query Phenotypes
- {'; '.join(patient_hpos)}\n
### Diagnostic Predictions from Four Different Models
**1. Deep Learning:**
{chr(10).join([f"   {no + 1}. {self.dise_name_map[diag_code.split(';')[-1]]} [{diag_code.split(';')[-1]}]" for no, diag_code in enumerate(rl_topn)])}\n
**2. Likelihood Ratio:**
{chr(10).join([f"   {no + 1}. {self.dise_name_map[diag_code.split(';')[-1]]} [{diag_code.split(';')[-1]}]" for no, diag_code in enumerate(lr_topn)])}\n
**3. LLM Zero-shot Diagnosis:**
{chr(10).join([f"   {no + 1}. {self.dise_name_map[diag_code.split(';')[-1]]} [{diag_code.split(';')[-1]}]" for no, diag_code in enumerate(llm_topn)])}\n
**4. PBR (Patient-Based Retrieval) with Detailed Evidence:**
*This method identifies similar historical cases to support the diagnosis.*
{pbr_string}\n
### The Robust Rank Aggregation Results of Above Four Models:
{chr(10).join([f"   {no + 1}. {self.dise_name_map[diag_code.split(';')[-1]]}" for no, diag_code in enumerate(rra_topn)])}\n
### Disease-related Phenotypic Information (Detailed Analysis) with Disease Code:
*Order based on Robust Rank Aggregation results.*
{diagnosis_detail_string}\n

Let's reason the Top {TOP_K} most likely diseases and present them at the vary end of your response in the agreed-upon JSON format:"""

        llm_ans = llm_query(msg, pheno_geno_analysis)
        llm_top_k = self.extract_json_results(llm_ans)
        llm_top_k = [self.orpha_omim2code[i.split(';')[0]] for i in llm_top_k]
        llm_top_k = list(dict.fromkeys([i if i not in missed_valid_mapping else missed_valid_mapping[i] for i in llm_top_k]))
        rest_rra_topn = [i for i in rra_topn if i not in llm_top_k]
        combined_top_n = (llm_top_k + rest_rra_topn)[:RECOMMEND_TOP_N]

        return combined_top_n, llm_ans

    def extract_json_results(self, model_output):
        if model_output is None:
            return None
        try:
            json_match = re.search(r'(\{.*?\})', model_output, re.DOTALL | re.MULTILINE)
            if json_match:
                json_str = json_match.group(1)
                json_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
                data = json.loads(json_str)
                results = [data.get(f"Rank {_topk + 1}") for _topk in range(TOP_K)]
                if all(results) and len(results) == TOP_K:
                    code_pattern = r"\b(?:OMIM|ORPHA):\d+\b"
                    final_codes = []
                    for seg in results:
                        match = re.search(code_pattern, seg, re.IGNORECASE)
                        if match:
                            final_codes.append(match.group().upper())
                        else:
                            final_codes.append(self.dise_code_map.get(seg, None))
                    return [_code for _code in final_codes if _code is not None]
        except Exception as e:
            print(f"JSON Parsing Error: {e}")
        return None


def llm_query(usr_msg, pheno_geno_analysis=False):
    gpt_url = "{}/chat/completions".format(OPENAI_API_BASE)
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer {}".format(OPENAI_API_KEY)
    }
    data = {
        "model": GPT_MODEL,
        "messages": [
            {"role": "system", "content": SYS_MSG_1 if not pheno_geno_analysis else SYS_MSG_2},
            {"role": "user", "content": usr_msg}
        ],
        "stream": False,
        "temperature": 0.3
    }
    for retry in range(3):
        try:
            response = json.loads(requests.post(gpt_url, headers=headers, json=data).text)
            return response['choices'][0]['message']['content']
        except Exception as e:
            time.sleep(1)
            continue
    return None
