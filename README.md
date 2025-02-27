# Evaluation of the gain/loss in geographic knowledge quality

Through 6 criteria:

- Model family (Llama, Mistral, Qwen)
- Model size (from 1B to 70B)
- Quantization level (int4, int8, float16, bi.float16)
- Fine-tuning (Base and Instruct models)
- Different prompts
- English and French languages

## Evaluation of geographic knowledge quality

2 evaluations proposed:

- Asking the model to predict the GPS coordinates of geographic areas
- Extracting place embeddings and matching geographic coordinates using linear regression

## Reproduce our work

1. Python virtual env

- Recrate the virtual env with conda: `conda env create -f requirements.yml`. It will create a conda env called `geo-llm`.
- Activate this env: `conda activate geo-llm`

2. Pipeline configuration

- Copy and adapt the config: `cp pipeline_config.default.py pipeline_config.py`
- Add your HuggingFace Access Token.
- Configure the list of models used. Make sur to have a granted access to their weights through [HuggingFace](huggingface.co/)

3. Run the pipeline

- Extract embedding and generate GPS coordinates with LLM: `python 1.embeddings.py` 
- Map GPS coordinates using embedding with linear regression : **TODO**

## Authors

| Auteur                                             | Institution                        |
|----------------------------------------------------|------------------------------------|
| [Adrien Guille](https://adrienguille.github.io/)   | Lyon 2, UR ERIC                    |
| [Rémy Decoupes](https://remy.decoupes.pages.mia.inra.fr/website/) | UMR TETIS / INRAE   |
