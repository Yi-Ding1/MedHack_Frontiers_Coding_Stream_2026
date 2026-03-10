## MedHack: Frontiers 2026 – Winning Solution (Small Track, Coding)

This repository contains the **first-place winning machine learning pipeline** developed by my team for the **MedHack: Frontiers 2026 – Small Track (Coding)** competition. The goal of the challenge is to **classify a patient’s health state at each timestamp** into one of **four clinical states**:

- **normal**
- **warning**
- **crisis**
- **death**

The model operates on **time-series clinical data** (e.g., vital signs and monitoring signals) and **static patient-level features** drawn from the competition datasets.

### Problem description

- **Task**: Multiclass sequence classification of patient health state over time.
- **Labels**: `normal`, `warning`, `crisis`, `death`, defined at each time step for each encounter.
- **Inputs**:
  - **Dynamic features** from longitudinal tabular data (e.g., vital signs and other timestamped measurements).
  - **Static patient attributes** from `patients.csv`, including:
    - Demographics (age, gender, race, ethnicity).
    - Socio-clinical context (marital_status, encounter_class, encounter_description, reason_for_visit).
    - Clinical history and status (previous_medical_history, current_medications, previous_medications, known_allergies, bmi, pain_score).

The pipeline combines these sources to estimate the evolving risk level of each patient throughout their encounter.

### What this repository includes

- **End-to-end data analysis (DSL documentation)**
  - `dslc_documentation/01_data_analysis.ipynb` contains the main exploratory analysis.
  - Examines dynamic vital-sign features in `train_data.csv`, including timestamp-level missingness patterns to motivate imputation and masking.
  - Analyzes static patient attributes in `patients.csv`, including missingness, categorical cardinalities, mutual information with downstream labels, correlation structure, and distributions (e.g., pain_score).
  - Inspects high-cardinality clinical text fields (previous medical history, medications, allergies) and their semicolon-separated structure to guide feature engineering.

- **Final competition pipeline**
  - Reproducible final notebook: `dslc_documentation/02_final_pipeline.ipynb`.
  - Deterministic training setup with fixed random seeds and controlled PyTorch backends.
  - Data preprocessing for longitudinal inputs (e.g., interpolation, forward/backward filling, consistent indexing across time).
  - Integration of **static patient features** with **time-series signals** at the encounter level.
  - Final modeling and prediction pipeline used for the winning submission, including preprocessing, model training, inference, and submission file generation.
  - Saves the trained model artifacts and holdout-set predictions used for the competition.

- **Supporting assets**
  - `data/README.txt` describing the competition data interface and access constraints.

### What is intentionally not included

To keep this repository focused on the **final, reproducible solution**, several experimental components from the competition are **not** included:

- **Extensive hyperparameter tuning** (to keep the code clean and easier to understand).
- **Full model architecture search** (alternative deep architectures and variants that were tried and discarded).
- **Intermediate experiments** such as:
  - Early baseline models.
  - Alternative feature encodings and text-processing strategies.
  - Side-by-side comparisons of different sequence models and ensembles.

Only the **final, competition-winning pipeline** and its essential analysis steps are preserved here.

### Data access and privacy

The raw data used in this solution comes from the **MedHack: Frontiers 2026** competition and is subject to **strict access and disclosure rules**:

- The data may **not** be publicly disclosed, redistributed, or uploaded outside the competition environment.
- This repository does **not** ship with the original competition datasets.
- Authorized participants should place the official data files (such as `train_data.csv`, `test_data.csv`, `holdout_data.csv`, `patients.csv`, and related files) into the `data/` directory, following the instructions in `data/README.txt`.

If you are viewing this repository outside the official competition context, assume that you will **not** have direct access to these datasets and must obtain them via the official MedHack channels if eligible.

### Environment setup

- **Python version**: This project has been tested with **Python 3.10+**.
- **Install dependencies from `requirements.txt` (CPU PyTorch by default)**:
  - Create and activate a virtual environment (recommended).
  - From the repository root (where `requirements.txt` lives), run:

    ```bash
    pip install -r requirements.txt
    ```

- **GPU-accelerated PyTorch (recommended)**:
  - The command above will typically install the **CPU-only** build of PyTorch from PyPI.
  - If you have a compatible NVIDIA GPU, install a **CUDA-enabled** PyTorch build that matches your GPU + driver/CUDA setup.
  - Use the official installer selector at [PyTorch Get Started](https://pytorch.org/get-started/locally/) to choose the right CUDA version and the exact pip command for your machine.
  - If you previously installed CPU-only PyTorch in the same environment, you may need to **remove the existing PyTorch packages first** before reinstalling the CUDA-enabled build.

- **Key libraries**: The requirements file includes `numpy`, `pandas`, `torch`, `scikit-learn`, `matplotlib`, and `seaborn`, which cover all non-standard dependencies used in the notebooks.

### Getting started

- **Review the data description**: Start with `data/README.txt` to understand the expected files and their purpose.
- **Explore the analysis**:
  - Open `dslc_documentation/01_data_analysis.ipynb` for a detailed, notebook-based exploration of dynamic vitals, static features, and clinical text fields.
- **Inspect the final pipeline**:
  - Open `dslc_documentation/02_final_pipeline.ipynb` to follow the full data preprocessing, model training, and inference flow used for the winning submission.

