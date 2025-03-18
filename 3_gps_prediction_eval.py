from pipeline_config import list_of_models, list_of_prompts, quantizations, mistral_models
import numpy as np
import pandas as pd
import datetime
import traceback
import pickle as pk
import re
import math

cities = pk.load(open("outputs/cities_embeddings.pk", "rb"))
cities = pk.load(open("outputs/TMP/TMP_cities_embeddings_Mistral-Small-24B-Base-2501_int4_gps_en.pk", "rb"))
cities[['Latitude', 'Longitude']] = cities['Coordinates'].str.split(", ", expand=True)
cities['Latitude'] = cities['Latitude'].astype(float)
cities['Longitude'] = cities['Longitude'].astype(float)

def dms_to_decimal(degrees, minutes, seconds, direction):
    """Convert DMS (Degrees, Minutes, Seconds) to Decimal format."""
    if seconds == None:
        seconds = '00'
    decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    if direction in ["S", "O", "W"]:  # South and West are negative
        decimal *= -1
    return decimal


def extract_coordinates(text):
    """
    Extrait la première latitude et longitude d'un texte donné.
    Prend en charge les formats décimaux et DMS.
    """
    
    # ✅ Regex pour les coordonnées en format Décimal
    decimal_pattern = re.search(r'''
        (?:latitude\s*[:de]*\s*)?([-+]?\d+\.\d+)°?\s*([NS])?  # Latitude (optionnellement avec N/S)
        (?:\s*[,;/]\s*|\s*(?:et\s*à\s*une\s*|longitude\s*[:de]*)\s*)  # Séparateurs possibles
        ([-+]?\d+\.\d+)°?\s*([EO])?  # Longitude (optionnellement avec E/O)
    ''', text, re.VERBOSE | re.IGNORECASE)

    if decimal_pattern:
        lat = float(decimal_pattern.group(1))
        lon = float(decimal_pattern.group(3))
        if decimal_pattern.group(2) and decimal_pattern.group(2).upper() == 'S':
            lat *= -1
        if decimal_pattern.group(4) and decimal_pattern.group(4).upper() in ['O', 'W']:
            lon *= -1
        return lat, lon

    # ✅ Regex pour les coordonnées en format DMS
    dms_pattern = re.search(r'''
        (?:(?:Latitude|latitude)\s*[:]*\s*)?(\d+)°\s*(\d+)['’′]?\s*(\d+)?["”″]?\s*([NSnordsud])  # Latitude
        .{0,20}?  # Gestion de texte intermédiaire variable
        (?:(?:Longitude|longitude)\s*[:]*\s*)?(\d+)°\s*(\d+)['’′]?\s*(\d+)?["”″]?\s*([EOestouest])  # Longitude
    ''', text, re.VERBOSE | re.IGNORECASE)

    if dms_pattern:
        lat = dms_to_decimal(dms_pattern.group(1), dms_pattern.group(2), dms_pattern.group(3), dms_pattern.group(4))
        lon = dms_to_decimal(dms_pattern.group(5), dms_pattern.group(6), dms_pattern.group(7), dms_pattern.group(8))
        return lat, lon

    return None, None  # Si aucune coordonnée trouvée


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth's surface.
    Parameters:
    lat1, lon1 -- Latitude and longitude of the first point in decimal degrees
    lat2, lon2 -- Latitude and longitude of the second point in decimal degrees
    Returns:
    distance -- Distance between the two points in kilometers
    """
    
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    # Radius of the Earth in kilometers (mean radius)
    R = 6371.0
    distance = R * c
    return distance

def calculate_distance(row, checkpoint, quantization, name):
    """
    Calculate the distance between the predicted and ground truth coordinates for a row.
    Parameters:
    row -- the row of the DataFrame
    checkpoint, quantization, name -- strings used to access predicted coordinates columns
    Returns:
    distance -- distance between predicted and ground truth in kilometers
    """
    # Extract predicted coordinates
    lat_pred = row[f"{checkpoint}_{quantization}_{name}_lat_predicted"]
    lon_pred = row[f"{checkpoint}_{quantization}_{name}_lon_predicted"]
    
    # Extract ground truth coordinates
    lat_truth = row['Latitude']
    lon_truth = row['Longitude']
    
    # Calculate distance using haversine function
    return haversine(lat_pred, lon_pred, lat_truth, lon_truth)

results = []
for CHECKPOINT in list_of_models:
    print(f" - {CHECKPOINT} - ")
    if "70B" in CHECKPOINT or "72B" in CHECKPOINT:
            quantization = "int4"
            if f"{CHECKPOINT}_{quantization}_gps_fr_output" in cities.columns:
                for name, PROMPT in list_of_prompts.items():
                    if "gps" in name:
                        cities[f"{CHECKPOINT}_{quantization}_{name}_lat_predicted"], cities[f"{CHECKPOINT}_{quantization}_{name}_lon_predicted"] = zip(*cities[f"{CHECKPOINT}_{quantization}_{name}_output"].apply(extract_coordinates))
                        cities[f"{CHECKPOINT}_{quantization}_{name}_distance"] = cities.apply(calculate_distance, axis=1, checkpoint=CHECKPOINT, quantization=quantization, name=name)
    else:
        for quantization in quantizations:
            if f"{CHECKPOINT}_{quantization}_gps_fr_output" in cities.columns:
                for name, PROMPT in list_of_prompts.items():
                    # print(f"{CHECKPOINT.split('/')[-1]}_{quantization}_{name}")
                    if "gps" in name:
                        cities[f"{CHECKPOINT}_{quantization}_{name}_lat_predicted"], cities[f"{CHECKPOINT}_{quantization}_{name}_lon_predicted"] = zip(*cities[f"{CHECKPOINT}_{quantization}_{name}_output"].apply(extract_coordinates))
                        cities[f"{CHECKPOINT}_{quantization}_{name}_distance"] = cities.apply(calculate_distance, axis=1, checkpoint=CHECKPOINT, quantization=quantization, name=name)

df = pd.DataFrame(cities)
date_str = datetime.datetime.now().strftime("%Y-%m-%d")
df.to_csv(f"outputs/cities_prediction_{date_str}.csv", index=False)
df.to_csv(f"outputs/cities_prediction.csv", index=False)
print(df.head())