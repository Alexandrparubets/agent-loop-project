import faiss
import numpy as np
import pickle


def build_faiss_index(embeddings):
    # FAISS работает с float32
    vectors = np.array(embeddings).astype("float32")

    dim = vectors.shape[1]

    # простой индекс (brute-force)
    index = faiss.IndexFlatL2(dim)

    index.add(vectors)

    return index


def save_index(index, path="data/index.faiss"):
    faiss.write_index(index, path)


def load_index(path="data/index.faiss"):
    return faiss.read_index(path)


def save_chunks(chunks, path="data/chunks.pkl"):
    with open(path, "wb") as f:
        pickle.dump(chunks, f)


def load_chunks(path="data/chunks.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)