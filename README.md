# 🔬 Data-Driven Analysis and Machine Learning for Solid Oxide Cell (SOC) Fabrication

*This project deals with the application of machine learning to the process of tape casting, which is one of the main manufacturing processes for solid oxide fuel and electrolyser cells. It further targets to predict the physical properties of the fuel-electrode support across stages such as tape casting, sintering, and reduction.*

![Python](https://img.shields.io/badge/python-3.10-3670A0?logo=python&logoColor=ffdd54)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🗂️ Repository Structure

```bash
ml-manufacturing-process-prediction/
│
├── README.md # This file
├── requirements.txt # Python dependencies
├── .gitignore # Ignored files and folders
│
├── data/
│ ├── raw/ # Original data (not included)
│ ├── processed/ # Cleaned data used for modelling (not included)
│ └── README.md # Data description and format
│
├── notebooks/
│ └── model_selection.ipynb # Notebook for model comparison and selection
│
├── src/
│ ├── __init__.py # Makes src a Python package
│ ├── main.py # Main pipeline: training and evaluation
│ ├── model_trainer.py # Functions to train and evaluate models
│ └── exploratory_data_analysis_utils.py # Helper functions for data analysis
│
└── outputs/
├── figures/ # Generated plots
└── models/ # Saved models (not included)
```
---

> **Note:** `data/` and `outputs/models/` are not included due to confidentiality and reproducibility. See folder-specific READMEs for details.

---

## 📊 Data

The datasets used in this project are **not publicly available**.  

**Data description:**

- Input variables: `tape_id`, `temperature`, `doctor_blade`, `casting_speed`, `humidity`, `volume_flow_rate`  

- Outputs: measured tape properties at green, sintered, and reduced stages  

**Expected format (for your own data):**

tape_id, temperature, doctor_blade, casting_speed, humidity, volume_flow_rate, green_thickness, green_density, sintered_thickness, sintered_density, reduced_thickness, reduced_density

1, 20, 200, 5, 50, 70, 66, 3.45, 63, 4.25, 62, 3.82

2, 25, 200, 5, 30, 60, 65, 3.42, 63, 4.2, 62, 3.8

...

Place your dataset in `data/processed/` and ensure column names match the scripts.

📄 See [`data/README.md`](data/README.md) for more information.

---

## 📈 Notebooks

`notebooks/model_selection.ipynb` includes:


- Multiple ML model training & tuning  

- Repeated k-fold cross-validation  

- Best model selection  

- Visualisations and model evaluation  

---

## 🖥 Python Code (`src/`)

- `model_trainer.py` – train, evaluate, and save ML models  

- `main.py` – main pipeline to train best model on full dataset and generate outputs  

- `exploratory_data_analysis_utils.py` – helper functions for data analysis  

Run the main pipeline:

```bash
python src/main.py
```

## 💾 Outputs

Figures: saved in outputs/figures/

Models: saved in outputs/models/ (not included to avoid large files and version conflicts; folder exists with .gitkeep)

Metadata (library versions, parameters) is recommended when saving models locally to ensure reproducibility.

## 🚀 Getting Started

1. Clone the repository:

```bash
git clone https://github.com/ledinhtan/ml-manufacturing-process-prediction.git
cd ml-manufacturing-process-prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```
3. Run the main pipeline:

```bash
python src/main.py
```

4. Explore the notebook:
```bash
jupyter notebook notebooks/model_selection.ipynb
```
---

## 👨‍💻 Author
Tan LE DINH

- GitHub: [@ledinhtan](https://github.com/ledinhtan)
- LinkedIn: [Lê Đình Tấn](https://www.linkedin.com/in/tan-le-dinh-25a863254/)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
