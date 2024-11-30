import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_local_llm():
    model_name="distilgpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval() 
    return model, tokenizer

def ask_llm(model, tokenizer, prompt, max_tokens=1024):
    """
    Generate a response from the model with support for longer outputs using looping.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    response_ids = input_ids

    for _ in range(3): 
        output = model.generate(
            response_ids,
            max_length=response_ids.size(1) + max_tokens,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=False,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )

        response_ids = torch.cat((response_ids, output[:, response_ids.size(1):]), dim=1)
        if response_ids.size(1) >= 2048:
            break

    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response
