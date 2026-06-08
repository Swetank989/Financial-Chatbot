import numpy as np

def search_chunks(query_embedding, index, chunks, k=3):

    distances, indices = index.search(
        np.array([query_embedding], dtype="float32"),
        k
    )

    results = [chunks[i] for i in indices[0]]

    return results