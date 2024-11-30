import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model_for_query(model_name="distilgpt2"):
    """
    Load the Hugging Face model and tokenizer for generating embeddings.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    return model, tokenizer

def query_faiss_index(query, index_file="embeddings/faiss_index"):
    """
    Query the FAISS index for matches based on the query.
    """
    model_name = "distilgpt2"
    model, tokenizer = load_model_for_query(model_name)

    index = faiss.read_index(index_file)
    with open(f"{index_file}_metadata.npy", "rb") as meta_file:
        metadata = np.load(meta_file, allow_pickle=True)

    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    distances, indices = index.search(np.array([query_embedding]), k=3)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        doc_name, sentence = metadata[idx]
        results.append((doc_name, sentence, dist))

    return results

if __name__ == "__main__":
    query = "What are the differences between Tesla and Uber?"
    results = query_faiss_index(query)
    for doc, sentence, dist in results:
        print(f"Document: {doc}\nMatch: {sentence}\nDistance: {dist}\n")
