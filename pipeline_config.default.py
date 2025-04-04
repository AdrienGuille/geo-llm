hf_access_token = "hf..."

mistral_models = ["Mistral-7B-v0.1", "Mistral-7B-Instruct-v0.1", 
                  "Mistral-7B-Instruct-v0.2", 
                  "Mistral-7B-v0.3", "Mistral-7B-Instruct-v0.3",
                  # "Mistral-Nemo-Base-2407", "Mistral-Nemo-Instruct-2407", # need a particular arg with transformers
                  "Mistral-Small-24B-Base-2501", "Mistral-Small-24B-Instruct-2501"]
mistral_models = [f"mistralai/{model}" for model in mistral_models]

llama_models = ["Llama-3.2-1B", "Llama-3.2-1B-Instruct", 
                   "Llama-3.2-3B", "Llama-3.2-3B-Instruct", 
                   "Llama-3.1-8B", "Llama-3.1-8B-Instruct", 
                   "Llama-2-7b-hf", "Llama-2-7b-chat-hf",
                   "Llama-2-13b-hf", "Llama-2-13b-chat-hf",
                   "Llama-3.1-70B", "Llama-3.1-70B-Instruct",]
llama_models = [f"meta-llama/{model}" for model in llama_models]

qwen_models = [
    "Qwen2.5-0.5B", "Qwen2.5-0.5B-Instruct",
    # "Qwen2.5-1.8B", "Qwen2.5-1.8B-Instruct", # NA anymore
    # "Qwen2.5-4B", "Qwen2.5-4B-Instruct", # NA anymore
    "Qwen2.5-7B", "Qwen2.5-7B-Instruct",
    "Qwen2.5-14B", "Qwen2.5-14B-Instruct",
    "Qwen2.5-32B", "Qwen2.5-32B-Instruct",
    "Qwen2.5-72B", "Qwen2.5-72B-Instruct"
]

qwen_models = [f"Qwen/{model}" for model in qwen_models]

list_of_models =  mistral_models + llama_models + qwen_models

list_of_prompts = {
    "empty":"", 
    "where_fr": "Où se trouve la ville de ", 
    "gps_fr": "Quelles sont les coordonnées géographiques de la ville de ", 
    "where_en": "Where is the city of ", 
    "gps_en": "What are the geographical coordinates of the city of "
}

quantizations = ["float16", "int8", "int4"]