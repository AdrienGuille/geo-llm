import pickle
import pandas as pd

# List of input pickle files
input_files = [
    "outputs/TMP/intermediate_results.pk",
    "outputs/TMP/TMP_cities_embeddings_Qwen2.5-72B-Instruct_int4_gps_en.pk"
]

# Output pickle file
output_file = "outputs/TMP/intermediate_results_2.pk"

def load_pickle(file_path):
    """Load a pickle file without modifying data types."""
    with open(file_path, "rb") as f:
        return pickle.load(f)

def merge_pickles_columnwise(input_files, output_file):
    """Merge multiple pickled Pandas DataFrames column-wise (preserving common columns)."""
    dataframes = [load_pickle(f) for f in input_files]

    # Ensure all loaded objects are Pandas DataFrames
    if not all(isinstance(df, pd.DataFrame) for df in dataframes):
        raise TypeError("Not all loaded objects are Pandas DataFrames.")

    # Merge on common columns (outer join to keep all columns)
    df_merged = dataframes[0]
    for df in dataframes[1:]:
        df_merged = df_merged.merge(df, on=list(set(df_merged.columns) & set(df.columns)), how="outer")

    # Save the merged DataFrame using pickle
    with open(output_file, "wb") as f:
        pickle.dump(df_merged, f)

    print(f"Successfully merged {len(input_files)} files column-wise into {output_file}")

# Run the merge function
merge_pickles_columnwise(input_files, output_file)