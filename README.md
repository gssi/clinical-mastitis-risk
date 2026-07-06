# Temporal modeling and explainability-based analysis for clinical mastitis risk in dairy cattle: A data-driven approach

## 1. Overview

This repository implements a modular data-driven workflow for binary classification of clinical mastitis risk in dairy cattle, with a specific focus on temporal modeling and explainability-based analysis.
The workflow supports both end-to-end experimentation and selective execution of individual components:

- Multi-source data integration
- Data preprocessing and domain-informed transformation
- Hierarchical missing-data imputation
- Temporal anchor construction
- Temporal sampling for balanced experimental datasets
- Coherence construction for controlled cross-technique comparison
- Machine Learning (ML) modeling with lagged tabular representation
- Explainability analysis with ML
- Deep Learning (DL) modeling with sequential tensor representation

The implementation reflects the methodology described in the accompanying paper and supports reproducible experimentation through explicit command-line interface and environment configuration.

---

## 2. Folder Structure

The repository is organized as a modular pipeline, where each component corresponds to a specific step of the workflow. Components can be executed independently or combined into an end-to-end process.

```text
clinical_risk_classification/
│
├── command_lines/                                      # Command lines for running the workflow. NOTE: They must be adapted to the workspace and/or to the goal of the user.
│   ├── commands_data_construction.txt                  # Commands for processing raw source tables and building the merged dataset
│   ├── commands_data_preprocessing.txt                 # Commands for transformation, imputation, temporal construction, sampling, and coherence construction
│   ├── commands_ml.txt                                 # Commands for ML lagged-tabular construction and training
│   └── commands_dl.txt                                 # Commands for DL tensor construction and training
│
├── workflow/                                           # Main workflow package
│   ├── data_preprocessing/                             # Data preprocessing and anchor construction modules
│   │   ├── coherence_construction.py                   # Provides the shared anchor set for ML/DL comparability
│   │   ├── imputation.py                               # Hierarchical imputation of missing values
│   │   ├── temporal_construction.py                    # Provides temporally valid anchors over lag windows
│   │   ├── temporal_sampling.py                        # Controlled data balancing
│   │   ├── transformation.py                           # Preprocessing and domain-informed transformations
│   │   └── dataset_construction/                       # Source-specific processing and dataset integration
│   │       ├── anagraphic_step.py                      # Processes demographic/anagraphic records
│   │       ├── calving_step.py                         # Processes calving records
│   │       ├── dataset_builder.py                      # Merges processed sources into the final longitudinal dataset
│   │       ├── diseases_step.py                        # Processes treatment/diagnosis records
│   │       ├── ele_conductivity_step.py                # Processes electrical conductivity records
│   │       ├── functional_check_step.py                # Processes functional control records
│   │       └── lactose_step.py                         # Processes lactose records
│   │
│   ├── ml_process/                                     # Machine Learning branch
│   │   ├── lagged_tabular_construction.py              # Builds the lagged tabular ML representation
│   │   └── training.py                                 # Trains ML models, (optionally) runs explainability, and provides classification results
│   │
│   └── dl_process/                                     # Deep Learning branch
│       ├── tensor_construction.py                      # Builds sequential tensor inputs for recurrent models
│       └── training.py                                 # Trains RNN/LSTM/GRU models and provides classification results
│
├── workspace/                                          # User-defined working directory (not included)
│   ├── data/                                           # Input and processed datasets (not included)
│   │   ├── db_modeling/                                # Input and processed data for modeling branches (not included)
│   │   │   ├── dl/                                     # Input and processed data for DL branch (not included)
│   │   │   ├── ml/                                     # Input and processed data for ML branch (not included)
│   │   │   ├── shared/                                 # Input and processed data common to both branches (not included)
│   ├── artifacts/                                      # Schemas, ids, metadata, fitted imputer (not included)
│   ├── logs/                                           # Reports and logs (not included)
│   └── models/                                         # Classification results and hyperparameter files (not included)
│
├── download_tables.py                                  # Utility script for downloading/source tables from LEO 
├── libraries.py                                        # Shared imports/utilities
├── merge_tables.py                                     # Utility script for merging tables downloaded from LEO
├── select_features.py                                  # Utility script for feature subset selection
└── requirements.yml                                    # Conda environment specification
```

---

## 3. Installation 

The project is built for a Unix-like shell. 

### Install Conda

This project uses Conda for dependency management. If not installed, install from: 

https://docs.conda.io/en/latest/miniconda.html

### Download project

Download the repository, then move into the project directory:

```bash
cd clinical_risk_classification
```

### Create the Conda environment

The project dependencies are defined in the `requirements.yml` file. To create the environment:

```bash
conda env create -f requirements.yml
```

### Activate the created environment

```bash
conda activate mastitis-framework
```

---

## 4. Usage

The repository includes ready-to-use command examples in the `command_lines/` folder. The provided commands are designed to be executed from the project root.  
All commands assume the following execution pattern:

```bash
PYTHONPATH=. python3 <script_path> [arguments]
```

### Execution for controlled comparison between ML and DL

The full pipeline is executed to ensure that both models are trained and evaluated on the same subject–time instances. 

1. Data construction → `command_lines/commands_data_construction.txt`
2. Transformation → `command_lines/commands_data_preprocessing.txt`
3. Imputation → `command_lines/commands_data_preprocessing.txt`
4. Temporal construction → `command_lines/commands_data_preprocessing.txt`
5. Temporal sampling → `command_lines/commands_data_preprocessing.txt`
6. Coherence construction → `command_lines/commands_data_preprocessing.txt`
7. ML branch → `command_lines/commands_ml.txt`
8. DL branch → `command_lines/commands_dl.txt`





