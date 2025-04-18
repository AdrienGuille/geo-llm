from pipeline_config import list_of_models, list_of_prompts, mistral_models
import pickle as pk
import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import numpy as np
import pandas as pd
import datetime
import traceback
import json

cities = pk.load(open("outputs/intermediate_results_4.pk", "rb"))
cities = cities.sample(frac=1.0, random_state=0)
cities[['Latitude', 'Longitude']] = cities['Coordinates'].str.split(", ", expand=True)
cities['Latitude'] = cities['Latitude'].astype(float)
cities['Longitude'] = cities['Longitude'].astype(float)


for N in [100]:
    results = []
    for CHECKPOINT in list_of_models:
            print("###########################################")
            print(f" - {CHECKPOINT} - ")
            if "70B" in CHECKPOINT or "72B" in CHECKPOINT:
                quantizations = ["int4"]
            else:
                quantizations = ["float16", "int8", "int4"]
            for quantization in quantizations:
                for name, PROMPT in list_of_prompts.items():
                    print(f"{CHECKPOINT.split('/')[-1]}_{quantization}_{name}")
                    embeddings = np.array([
                        emb.numpy() if hasattr(emb, 'numpy') else np.nan  
                        for emb in cities[f"{CHECKPOINT}_{quantization}_{name}"]
                        if emb is not None and not (isinstance(emb, float) and np.isnan(emb))  # Filtrer NaN et None
                    ])
                    embeddings_new = []
                    nan_ids = []
                    for i in range(1000):
                        if sum(np.isnan(embeddings[i])) == 0:
                            embeddings_new.append(embeddings[i])
                        else:
                            nan_ids.append(i)
                    embeddings = np.array(embeddings_new)
                    print(embeddings.shape)
                    scaler = StandardScaler(with_mean=False, with_std=True)
                    embeddings = scaler.fit_transform(embeddings)
                    scores_r2 = {}
                    scores_mse = {}
                    predicted_coordinates = {}
                    for coord in ['Latitude', 'Longitude']:
                        regression = RidgeCV(alphas=[1e-1, 1, 5, 25, 50, 100, 250, 500, 750, 1500], store_cv_results=True)
                        regression.fit(embeddings[:N], cities[coord][:N])
                        pred_ids = [i for i in list(range(N,1000)) if i not in nan_ids]
                        scores_r2[coord] = r2_score(cities[coord][pred_ids], regression.predict(embeddings[N:1000-len(nan_ids)]))
                        scores_mse[coord] = mean_squared_error(cities[coord][pred_ids], regression.predict(embeddings[N:1000-len(nan_ids)]))
                        predicted_coordinates[coord] = regression.predict(embeddings[N:1000-len(nan_ids)])
                    distances = []
                    for true_lat, true_lon, pred_lat, pred_lon in zip(cities[N:]['Latitude'], cities[N:]['Longitude'], predicted_coordinates['Latitude'], predicted_coordinates['Longitude']):
                        truth_in_radians = [radians(_) for _ in [true_lat, true_lon]]
                        prediction_in_radians = [radians(_) for _ in [pred_lat, pred_lon]]
                        result = haversine_distances([truth_in_radians, prediction_in_radians])
                        distance_in_km = result[0,1] * 6371000/1000  # multiply by Earth radius to get kilometers
                        distances.append(distance_in_km)
                    model_type = "instruct" if ("Instruct" in CHECKPOINT or "chat" in CHECKPOINT) else "base"
                    results.append({"checkpoint": f"{CHECKPOINT.split('/')[-1]}_{quantization}_{name}",
                                    "type": model_type, 
                                    "quantization": quantization,
                                    "prompt": name,
                                    "r2_lat": scores_r2["Latitude"], 
                                    "r2_lng": scores_r2["Longitude"],
                                    "mse_lat": scores_mse["Latitude"], 
                                    "mse_lng": scores_mse["Longitude"],
                                    "distances_in_km": json.dumps(distances)})
    print("###########################################")
    df = pd.DataFrame(results)
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    df.to_csv(f"outputs/cities_regression_{date_str}_{N}.csv", index=False)
    df.to_csv(f"outputs/cities_regression.csv", index=False)
    #print(df)