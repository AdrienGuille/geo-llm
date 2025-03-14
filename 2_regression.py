from pipeline_config import list_of_models, list_of_prompts, quantizations, mistral_models

import pickle as pk
import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd
import datetime
import traceback

cities = pk.load(open("outputs/cities_embeddings.pk", "rb"))
# cities = pd.read_csv("outputs/TMP/TMP_cities_embeddings_Mistral-Small-24B-Base-2501_int4_gps_en.csv")
cities = pk.load(open("outputs/TMP/TMP_cities_embeddings_Mistral-Small-24B-Base-2501_int4_gps_en.pk", "rb"))
# cities = pd.read_csv("outputs/cities_embeddings.csv")
cities = cities.sample(frac=1.0, random_state=0)
cities[['Latitude', 'Longitude']] = cities['Coordinates'].str.split(", ", expand=True)
cities['Latitude'] = cities['Latitude'].astype(float)
cities['Longitude'] = cities['Longitude'].astype(float)

list_of_models =  mistral_models

N = 300
results = []
for CHECKPOINT in list_of_models:
    try:
        print("###########################################")
        print(f" - {CHECKPOINT} - ")
        if "70B" in CHECKPOINT or "72B" in CHECKPOINT:
                quantization = "int4"
                for name, PROMPT in list_of_prompts.items():
                    print(f"{CHECKPOINT.split('/')[-1]}_{quantization}_{name}")
                    embeddings = np.array([emb.numpy() for emb in cities[f"{CHECKPOINT}_{quantization}_{name}"]])
                    scaler = StandardScaler(with_mean=False, with_std=True)
                    embeddings = scaler.fit_transform(embeddings)
                    print(embeddings.shape[1])
                    scores = {}
                    for coord in ['Latitude', 'Longitude']:
                        regression = RidgeCV(alphas=[1e-1, 1, 5, 25, 50, 100, 250, 500, 750, 1500], store_cv_results=True)
                        regression.fit(embeddings[:N], cities[coord][:N])
                        scores[coord] = r2_score(cities[coord][N:], regression.predict(embeddings[N:]))
                    model_type = "instruct" if ("Instruct" in CHECKPOINT or "chat" in CHECKPOINT) else "base"
                    results.append({"checkpoint": f"{CHECKPOINT.split('/')[-1]}_{quantization}_{name}",
                                    "type": model_type, 
                                    "quantization": quantization,
                                    "prompt": name,
                                    "r2_lat": scores["Latitude"], 
                                    "r2_lng": scores["Longitude"]})
        else:
            for quantization in quantizations:
                for name, PROMPT in list_of_prompts.items():
                    # print(f"{CHECKPOINT.split('/')[-1]}_{quantization}_{name}")
                    embeddings = np.array([emb.numpy() for emb in cities[f"{CHECKPOINT}_{quantization}_{name}"]])
                    scaler = StandardScaler(with_mean=False, with_std=True)
                    embeddings = scaler.fit_transform(embeddings)
                    # print(embeddings.shape[1])
                    scores = {}
                    for coord in ['Latitude', 'Longitude']:
                        regression = RidgeCV(alphas=[1e-1, 1, 5, 25, 50, 100, 250, 500, 750, 1500], store_cv_results=True)
                        regression.fit(embeddings[:N], cities[coord][:N])
                        scores[coord] = r2_score(cities[coord][N:], regression.predict(embeddings[N:]))
                    model_type = "instruct" if ("Instruct" in CHECKPOINT or "chat" in CHECKPOINT) else "base"
                    results.append({"checkpoint": f"{CHECKPOINT.split('/')[-1]}_{quantization}_{name}",
                                    "type": model_type, 
                                    "quantization": quantization,
                                    "prompt": name,
                                    "r2_lat": scores["Latitude"], 
                                    "r2_lng": scores["Longitude"]})

    except Exception as e:
        print("###########################################")
        print(f" - Could not run exp with: {CHECKPOINT} - ")
        print(f"An error occurred: {e}")
        traceback.print_exc()
        print("###########################################")
print("###########################################")
df = pd.DataFrame(results)
date_str = datetime.datetime.now().strftime("%Y-%m-%d")
df.to_csv(f"outputs/cities_regression_{date_str}.csv", index=False)
df.to_csv(f"outputs/cities_regression.csv", index=False)
print(df.head())