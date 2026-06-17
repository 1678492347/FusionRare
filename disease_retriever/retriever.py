import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import *


def prepare_embeddings_list(embedded_documents):
    """Converts the embedded documents into a NumPy array of embeddings."""
    embeddings_list = [np.array(doc['embedding']) for doc in embedded_documents if
                       isinstance(doc['embedding'], np.ndarray) and doc['embedding'].size > 0]
    if not embeddings_list:
        exit(1)  # Exit if no valid embeddings found
    first_embedding_size = embeddings_list[0].shape[0]  # Ensure uniform embedding size
    return np.vstack([emb for emb in embeddings_list if emb.shape[0] == first_embedding_size])


def create_faiss_index_cosine(embeddings_array):
    """
    Creates a FAISS index for efficient cosine similarity searching.
    This is achieved by normalizing the vectors and using an Inner Product index.
    """
    if embeddings_array.dtype != np.float32:
        embeddings_array = embeddings_array.astype(np.float32)

    faiss.normalize_L2(embeddings_array)

    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_array)
    return index


print('Initializing Disease embedding')
embedding_model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=HF_CACHE_FOLDER)
embedded_disease_documents = np.load(DISEASE_NAME_EMB_PATH, allow_pickle=True)
embedded_disease_faiss_index = create_faiss_index_cosine(prepare_embeddings_list(embedded_disease_documents))
print('Disease FAISS created')
