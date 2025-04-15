# Evaluation of the gain/loss in geographic knowledge quality

The goal is to assess the impact of model size, quantization level, prompt engineering, language of the prompts, and standard fine-tuning on the quality of geographic knowledge within the LLMs.

Through 6 criteria:

| **Criterion**             | **Description**                                                  |
|---------------------------|------------------------------------------------------------------|
| **Model family**          | Llama, Mistral, Qwen                                             |
| **Model size**            | From 1B to 70B                                                   |
| **Quantization level**    | int4, int8, float16 - [[see HuggingFace](https://huggingface.co/docs/optimum/v1.17.1/concept_guides/quantization)]                                  |
| **Fine-tuning**           | Base and Instruct models                                         |
| **Different prompts**     | Various prompts to evaluate models' responses   [* See below]    |
| **Languages**             | English and French                                               |

[*] Example of prompts. You can configure them in [pipeline_config.py](https://github.com/AdrienGuille/geo-llm/blob/main/pipeline_config.default.py#L27):

```python
list_of_prompts = {
    "empty":"", 
    "where_fr": "Où se trouve la ville de ", 
    "gps_fr": "Quelles sont les coordonnées géographiques de la ville de ", 
    "where_en": "Where is the city of ", 
    "gps_en": "What are the geographical coordinates of the city of "
}
```

## Evaluation of geographic knowledge quality

2 evaluations proposed:

- Asking the model to predict the GPS coordinates of geographic areas
- Extracting place embeddings and matching geographic coordinates using linear regression

## Reproduce our work

1. Python virtual env

- Recrate the virtual env with conda: `conda env create -f requirements.yml`. It will create a conda env called `geo-llm`.
- Activate this env: `conda activate geo-llm`

2. Download input data

- Download France geonames data from OpenDataSoft: [[link](https://public.opendatasoft.com/explore/dataset/geonames-all-cities-with-a-population-1000/export/?flg=fr-fr&disjunctive.cou_name_en&sort=name&refine.cou_name_en=France)]

3. Pipeline configuration

- Copy and adapt the config: `cp pipeline_config.default.py pipeline_config.py`
- Add your HuggingFace Access Token.
- Configure the list of models used. Make sur to have a granted access to their weights through [HuggingFace](huggingface.co/)

4. Run the pipeline

- Extract embedding and generate GPS coordinates with LLM: `python 1.embeddings.py` 
- Map GPS coordinates using embedding with linear regression: `2_regression.py`
- Extract GPS coordinates from models' output: `3_gps_prediction_eval.py`
- Analyse results : `4_results_analyzing.ipynb`

5. postprocessing

- If you have to stop `python 1.embeddings.py` and re-run several times, you'll have to merge results into a single file: `src/postprocessing/concatenate_multiple_pickle.py`
- make the shifted maps (between reel GPS coordinates and predicted one): `src/postprocessing/shifted_maps.ipynb`

---

| Authors                                            |                         |
|----------------------------------------------------|------------------------------------|
| [Adrien Guille](https://adrienguille.github.io/)   | Lyon 2, UR ERIC                    |
| [Rémy Decoupes](https://remy.decoupes.pages.mia.inra.fr/website/) | UMR TETIS / INRAE   |
