from pipeline_config import hf_access_token, list_of_models, list_of_prompts, quantizations

from huggingface_hub import login; login(token=hf_access_token)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd
import pickle as pk
import torch

cities = pd.read_csv("data/cities.csv")
for CHECKPOINT in list_of_models:
    # try:
    print("###########################################")
    print(f" - {CHECKPOINT} - ")
    # Chargement du modèle
    try:
        tokenizer = AutoTokenizer.from_pretrained(f"{CHECKPOINT}")
    except:
        print("GPTQ tokenizer has to be loaded")
    model = None
    torch.cuda.empty_cache()
    if "70B" in CHECKPOINT : # LLMs > 70B need to be loaded in int4 on A100 80Go VRAM
        model = AutoModelForCausalLM.from_pretrained(f"{CHECKPOINT}",
                return_dict=True,
                torch_dtype=torch.float16,
                device_map="auto",
                quantization_config=BitsAndBytesConfig(load_in_4bit=True))
        quantization = "int4"
        for name, PROMPT in list_of_prompts.items():
            # Extraction des embeddings
            ENTITY_INDEX = -1
            embeddings = []
            outputs = []
            for city in cities.city:
                input = tokenizer(PROMPT+city, add_special_tokens=False, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    output = model(**input, output_hidden_states=True, output_attentions=False, return_dict=True, use_cache=False)
                    last_hidden_states = output.hidden_states[-1]
                    embeddings.append(last_hidden_states[0, ENTITY_INDEX].to("cpu"))
                    if "gps" in name:
                        predictions = model.generate(
                            **input,
                            max_length=256,
                            pad_token_id=tokenizer.pad_token_id,
                            temperature=0.3,
                            # top_p=0.9,
                            # top_k=50,
                            return_dict_in_generate=True,
                            output_scores=True,
                            pad_token_id=tokenizer.eos_token_id,
                            # repetition_penalty=1.2
                        )
                        generated_tokens = predictions.sequences.cpu()  # This contains the token IDs
                        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                    else: # no generative
                        generated_text = ""
                    outputs.append(generated_text)
                    del input, output, last_hidden_states
            cities[f"{CHECKPOINT}_{quantization}_{name}"] = embeddings
            cities[f"{CHECKPOINT}_{quantization}_{name}_output"] = outputs
    elif "GPTQ" in CHECKPOINT:
        tokenizer = AutoTokenizer.from_pretrained(f"{CHECKPOINT}")
        model = AutoModelForCausalLM.from_pretrained(CHECKPOINT,
                    return_dict=True,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto")
        quantization = "GPT-Q"
        for name, PROMPT in list_of_prompts.items():
            # Extraction des embeddings
            ENTITY_INDEX = -1
            embeddings = []
            outputs = []
            for city in cities.city:
                input = tokenizer(PROMPT+city, add_special_tokens=False, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    output = model(**input, output_hidden_states=True, output_attentions=False, return_dict=True, use_cache=False)
                    last_hidden_states = output.hidden_states[-1]
                    embeddings.append(last_hidden_states[0, ENTITY_INDEX].to("cpu"))
                    if "gps" in name:
                        predictions = model.generate(
                            **input,
                            max_length=256,
                            pad_token_id=tokenizer.pad_token_id,
                            temperature=0.3,
                            # top_p=0.9,
                            # top_k=50,
                            return_dict_in_generate=True,
                            output_scores=True,
                            pad_token_id=tokenizer.eos_token_id,
                            # repetition_penalty=1.2
                        )
                        generated_tokens = predictions.sequences.cpu()  # This contains the token IDs
                        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                    else: # no generative
                        generated_text = ""
                    outputs.append(generated_text)
                    del input, output, last_hidden_states
            cities[f"{CHECKPOINT}_{quantization}_{name}"] = embeddings
            cities[f"{CHECKPOINT}_{quantization}_{name}_output"] = outputs
    else:
        for quantization in quantizations:
            print(f"\t |->{quantization}")
            if quantization == "float16":
                model = AutoModelForCausalLM.from_pretrained(f"{CHECKPOINT}",
                                    return_dict=True,
                                    torch_dtype=torch.float16,
                                    device_map="auto")
            elif quantization == "int8":
                model = AutoModelForCausalLM.from_pretrained(f"{CHECKPOINT}",
                        return_dict=True,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        quantization_config=BitsAndBytesConfig(load_in_8bit=True))
            elif quantization == "int4":
                model = AutoModelForCausalLM.from_pretrained(f"{CHECKPOINT}",
                        return_dict=True,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        quantization_config=BitsAndBytesConfig(load_in_4bit=True))
            for name, PROMPT in list_of_prompts.items():
                print(f"\t\t |->{name}")
                ENTITY_INDEX = -1
                embeddings = []
                outputs = []
                torch.cuda.empty_cache()
                # Extraction des embeddings
                for city in cities.city:
                    input = tokenizer(PROMPT+city, add_special_tokens=False, return_tensors="pt").to("cuda")
                    with torch.no_grad():
                        output = model(**input, output_hidden_states=True, output_attentions=False, return_dict=True, use_cache=False)
                        last_hidden_states = output.hidden_states[-1]
                        embeddings.append(last_hidden_states[0, ENTITY_INDEX].to("cpu"))
                        if "gps" in name:
                            predictions = model.generate(
                                **input,
                                max_length=256,
                                pad_token_id=tokenizer.pad_token_id,
                                temperature=0.3,
                                # top_p=0.9,
                                # top_k=50,
                                return_dict_in_generate=True,
                                output_scores=True,
                                pad_token_id=tokenizer.eos_token_id,
                                # repetition_penalty=1.2
                            )
                            generated_tokens = predictions.sequences.cpu()  # This contains the token IDs
                            generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                        else: # no generative
                            generated_text = ""
                        outputs.append(generated_text)
                        del input, output, last_hidden_states
                cities[f"{CHECKPOINT}_{quantization}_{name}"] = embeddings
                cities[f"{CHECKPOINT}_{quantization}_{name}_output"] = outputs
                
                df = pd.DataFrame(cities)
                df.to_csv(f"outputs/TMP/TMP_cities_embeddings_{CHECKPOINT.split('/')[-1]}_{quantization}_{name}.csv")
                pk.dump(cities, open(f"outputs/TMP/TMP_cities_embeddings_{CHECKPOINT.split('/')[-1]}_{quantization}_{name}.pk", "wb"))
    # except:
    #     print("###########################################")
    #     print(f" - Could not run exp with: {CHECKPOINT} - ")
    #     print("###########################################")

df.to_csv(f"outputs/cities_embeddings.csv")        
pk.dump(cities, open(f"outputs/cities_embeddings.pk", "wb"))