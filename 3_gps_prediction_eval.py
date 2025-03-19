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

global_pattern_not_recognized_debug = []
global_output_truncated = 0

def extract_coordinates(text):
    """
    Extracts the first latitude and longitude from a given text.  
    Supports both decimal and DMS formats.

    DMS with clear latitude and longitude text:
        43° 16′ 00″ N, 5° 20′ 00″ E
        48° 51′ 24″ N, 2° 21′ 03″ E

    DMS with "Latitude" and "Longitude" explicitly mentioned:
        Latitude: 43° 21' 0" N, Longitude: 5° 22' 0" E

    DMS with values in parentheses:
        Latitude: 47°15'0"N (47.2500), Longitude: 2°10'0"W (-2.1667)

    Standard decimal degrees:
        48.8566°N latitude / 2.3522°E longitude.
        43.299723 latitude, 5.374999 longitude.
        45.766667 lat; 4.833333 long.
        14.2766700, 4.4000000

    Mixed or separated formats:
        Latitude: 43.299722, Longitude: 5.370278
        Latitude: 47°15'0"N, Longitude: 2°10'0"W
        """
    
    decimal_pattern = re.search(r'''
        (?:latitude\s*[:de]*\s*)?([-+]?\d+\.\d+)°?\s*(?:degrees|degrés)?\s*(North|South|Nord|Sud|N|S)?(?:\s*of\s*the\s*equator)?\.?\s*(?:[de]*\s*Latitude)?   # Lat
        (?:\s*[,;/]\s*|\s*(?:et\s*à\s*une\s*|et\s*à\s*la\s*|\s*and\s*)?(?:longitude\s*[:de]*)?\s*)?  # handle intermediate char*
        ([-+]?\d+\.\d+)°?\s*(?:degrees|degrés)?\s*(East|West|Est|Ouest|E|O)?\.?\s*(?:Longitude)?(?:\s*of\s*Prime\s*Meridian)?  # long
    ''', text, re.VERBOSE | re.IGNORECASE)

    if decimal_pattern:
        lat = float(decimal_pattern.group(1))
        lon = float(decimal_pattern.group(3))
        if decimal_pattern.group(2) and decimal_pattern.group(2).upper() == 'S':
            lat *= -1
        if decimal_pattern.group(4) and decimal_pattern.group(4).upper() in ['O', 'W']:
            lon *= -1
        return lat, lon

    # dms_pattern = re.search(r'''
    #     (?:(?:Latitude|latitude)\s*[:]*\s*)?(\d+)°\s*(\d+)['’′]?\s*(\d+)?["”″]?\s*([NSnordsud])  # Lat
    #     .{0,20}?  # handle intermediate char*
    #     (?:(?:Longitude|longitude)\s*[:]*\s*)?(\d+)°\s*(\d+)['’′]?\s*(\d+)?["”″]?\s*([EOestouest])  
    # ''', text, re.VERBOSE | re.IGNORECASE) # long
    # if dms_pattern:
    #     lat = dms_to_decimal(dms_pattern.group(1), dms_pattern.group(2), dms_pattern.group(3), dms_pattern.group(4))
    #     lon = dms_to_decimal(dms_pattern.group(5), dms_pattern.group(6), dms_pattern.group(7), dms_pattern.group(8))
    #     return lat, lon
        
    dms_pattern = re.search(r'''
        (?:(?:Latitude|latitude)\s*[:]*\s*)?(\d+)°\s*(\d+)['’′]?\s*(?:([\d.]+)?["”″]?)?\s*(?:Nord|Sud|N|S)  # Latitude avec option secondes
        .{0,20}?  # Gérer l'espace intermédiaire
        (?:(?:Longitude|longitude)\s*[:]*\s*)?(\d+)°\s*(\d+)['’′]?\s*(?:([\d.]+)?["”″]?)?\s*(?:Est|Ouest|E|O)  # Longitude
    ''', text, re.VERBOSE | re.IGNORECASE)
    if dms_pattern:
        lat_deg, lat_min, lat_sec, lon_deg, lon_min, lon_sec = dms_pattern.groups()
        
        # Convertir les valeurs en float
        lat = float(lat_deg) + float(lat_min) / 60 + (float(lat_sec) / 3600 if lat_sec else 0)
        lon = float(lon_deg) + float(lon_min) / 60 + (float(lon_sec) / 3600 if lon_sec else 0)

        return lat, lon
    elif "0000000000000000000000000000000000000000000" in text or "3333333333333333333333" in text:
        global global_output_truncated
        global_output_truncated = global_output_truncated + 1
    else:
        global_pattern_not_recognized_debug.append(text)


    return None, None 


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth's surface.
    Parameters:
    lat1, lon1 -- Latitude and longitude of the first point in decimal degrees
    lat2, lon2 -- Latitude and longitude of the second point in decimal degrees
    Returns:
    distance -- Distance between the two points in kilometers
    """
    
    if lon1 == None:
        return np.NaN

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
for pattern in global_pattern_not_recognized_debug:
    print(pattern)
    print("\n")
print(f"Nb of non-recognized pattern: {len(global_pattern_not_recognized_debug)}")
print(f"Nb of truncated prediction like: 'Latitude: 45.750000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000' : {global_output_truncated}")