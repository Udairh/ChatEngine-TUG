import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

def load_model_for_embeddings(model_name="distilgpt2"):
    """
    Load the Hugging Face model and tokenizer for generating embeddings.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    model.eval() 
    return model, tokenizer

def generate_embeddings(documents, index_file="embeddings/faiss_index"):
    """
    Generate embeddings for the documents and save them to the FAISS index.
    """
    embeddings_dir = os.path.dirname(index_file)
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)

    model_name = "distilgpt2" 
    model, tokenizer = load_model_for_embeddings(model_name)

    index = faiss.IndexFlatL2(768) 

    embeddings = []
    metadata = []

    for doc_name, content in documents.items():
        sentences = content.split('. ')
        
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)

            embedding = outputs.logits.mean(dim=1).squeeze().numpy()
            embeddings.append(embedding)
            metadata.append((doc_name, sentence))

    embeddings = np.array(embeddings)
    index.add(embeddings)

    faiss.write_index(index, index_file)
    with open(f"{index_file}_metadata.npy", "wb") as meta_file:
        np.save(meta_file, np.array(metadata, dtype=object))

    print(f"Embeddings and FAISS index have been saved to {index_file}")

if __name__ == "__main__":
    from parse_pdfs import parse_pdfs
    pdf_files = ["data/goog.pdf", "data/tesla.pdf", "data/uber.pdf"]
    documents = parse_pdfs(pdf_files)
    generate_embeddings(documents)
