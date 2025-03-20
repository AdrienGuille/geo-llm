import pandas as pd

# Define the list of input pickle files (modify as needed)
input_files = [
    "outputs/TMP/TMP_cities_embeddings_Qwen2.5-32B_float16_gps_fr.pk",
    "outputs/TMP/TMP_cities_embeddings_Mistral-Small-24B-Base-2501_int4_gps_en.pk"
]

# Define the output file
output_file = "outputs/TMP/intermediate_results.pk"

def merge_pickles(file_list, output_path):
    """Merge multiple pickle files containing Pandas DataFrames into one."""
    dataframes = [pd.read_pickle(f) for f in file_list]
    df_combined = pd.concat(dataframes, ignore_index=True)
    df_combined.to_pickle(output_path)
    print(f"Successfully merged {len(file_list)} files into {output_path}")

# Run the merging process
merge_pickles(input_files, output_file)