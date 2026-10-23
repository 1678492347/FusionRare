"""
##################################
General Configuration
##################################
"""
DATASET = ["OMIM", "ORPHA"]
LR_PRIOR_TYPE = "lr"
BATCH_SIZE = 32
RESIDUAL_HIDDEN_SIZE = 1024
JACCARD_SIM_THOD = 0.4  # Use a smaller JACCARD_SIM_THOD during testing
# JACCARD_SIM_THOD = 0.9 # A higher JACCARD_SIM_THOD can be used during deployment
EMBEDDING_MODEL = "FremyCompany/BioLORD-2023"

"""data files"""
HP_OBO_PATH = "data/hp.obo"                         # hp.obo PATH
PHENOTYPE_HPOA_PATH = "data/phenotype.hpoa"         # phenotype.hpoa PATH
DISEASE_NAME_PATH = "data/dise_synonyms.json"       # RD2name PATH
DISEASE_NAME_EMB_PATH = "data/dise_synonyms_BioLORD_metadata.npy"  # RD embedding PATH
ANNOTATION_FILE_PATH = "data/integrated_anno.json"  # RD phenotype annotation PATH
PROPAGATED_ANNO_PATH = "data/propagated_freq.json"  # RD propgated phenotype annotation PATH
LR_PRIOR_PATH = "data/integrated_lr.json"           # calculated LR PATH
RNC_CHECKPOINT = "data/checkpoint/rnc.pth"          # RNC weights
RPB_SAMPLE = "data/RPB_sample.json"                 # 1000 RPB samples

"""
##################################
Local Custom Configuration
##################################
"""
device = ""
GPT_MODEL = ""              # For best performance, use GPT‑5.5 for source diagnosis and reflection
OPENAI_API_BASE = ""        # API base
OPENAI_API_KEY = ""         # API Key
HF_CACHE_FOLDER = ""        # HuggingFace cache folder
RECOMMEND_TOP_N = 5         # Recommend top n disease, e.g., 5, 10
