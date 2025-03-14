import os
import glob
from datetime import datetime

repertoire = "outputs/TMP/" 
csv_files = glob.glob(os.path.join(repertoire, "*Llama*.csv"))

# ls -l | sort
file_timestamps = [(file, os.path.getmtime(file)) for file in csv_files]
file_timestamps.sort(key=lambda x: x[1])

print("Processing time between CSV file creations:\n")

elapsed_times = []
prev_file, prev_time = None, None
for file, timestamp in file_timestamps:
    file_time = datetime.fromtimestamp(timestamp)
    
    if prev_file:
        elapsed_time = file_time - prev_time
        # print(f"{prev_file} → {file}: {elapsed_time}")
        print(f"{file}: {elapsed_time}")
        elapsed_times.append((file, elapsed_time))
    
    prev_file, prev_time = file, file_time

# sort by descending processing time
print("\n")
print("Descending processing time: ")
elapsed_times.sort(key=lambda x: x[1], reverse=True)
for file, time_diff in elapsed_times:
    print(f"{file.replace('outputs/TMP/TMP_cities_embeddings_', '')}: {time_diff}")