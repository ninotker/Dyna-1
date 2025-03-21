# Datasets

We have additionally made the datasets available on 🤗HuggingFace at <a href='https://huggingface.co/datasets/gelnesr/RelaxDB'>gelnesr/Dyna-1</a>. 

We recommend downloading them using `huggingface-cli` by running the following commands on the command line:
```
pip install -U "huggingface_hub[cli]"
huggingface-cli download gelnesr/RelaxDB --repo-type dataset
```

In this repository on GitHub, you can find:
- `RelaxDB_pkls_22jan2025.zip`: unzip to access .pkl files (one per protein) for RelaxDB and RelaxDB-CPMG datasets.
- `RelaxDB_datasets/`: contains data in .json files for RelaxDB and RelaxDB-CPMG datasets.
- `probs`: contains saved frequencies from mBMRB-Train, stored for calculating dummy baselines.
- `split_files`: contains split files read in for train and evaluation datasets.
