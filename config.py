"""
#################
General Configuration
#################
"""
DATASET = ["OMIM"]
# DATASET = ["ORPHA"]
LR_PRIOR_TYPE = "lr"
BATCH_SIZE = 32
RESIDUAL_HIDDEN_SIZE = 1024
JACCARD_SIM_THOD = 0.4  # Use a smaller JACCARD_SIM_THOD during testing
# JACCARD_SIM_THOD = 0.9 # A higher JACCARD_SIM_THOD can be used during deployment
EMBEDDING_MODEL = "FremyCompany/BioLORD-2023"

"""
#################
Local Custom Configuration
#################
"""
device = ""
GPT_MODEL = ""          # For best performance, use GPT‑5.5 for source diagnosis and reflection
OPENAI_API_BASE = ""    # API base
OPENAI_API_KEY = ""     # API Key
HF_CACHE_FOLDER = ""    # HuggingFace cache folder
HP_OBO_PATH = ""            # hp.obo PATH
PHENOTYPE_HPOA_PATH = ""    # phenotype.hpoa PATH
DISEASE_NAME_PATH = ""      # RD2name PATH
DISEASE_NAME_EMB_PATH = ""  # RD name embedding PATH
ANNOTATION_FILE_PATH = ""   # RD phenotype annotation PATH
PROPAGATED_ANNO_PATH = ""   # RD propgated phenotype annotation PATH
LR_PRIOR_PATH = ""          # calculated LR PATH
RNC_CHECKPOINT = ""         # Trained RNC, e.g., checkpoint/rnc.pth

RECOMMEND_TOP_N = ""        # Recommend top n disease, e.g., 5, 10
