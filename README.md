# ESN-WiSARD: Resource-Efficient Time Series Classification

> **Academic Context**: This repository contains the official codebase and experimental benchmark developed as part of my **Bachelor's Thesis in Computer Science** (*Laurea Triennale in Informatica*). The project investigates lightweight, resource-efficient algorithms for time series classification on constrained devices.

Official repository for the **ESN-WiSARD** architecture, a Time Series Classification (TSC) framework combining Reservoir Computing (**Echo State Networks**) and Weightless Neural Networks (**WiSARD**) to achieve zero-FLOPs inference.

---

## Repository Structure

* **`main_pipeline.py`**: 
  The primary script executing the end-to-end classification pipeline and generating final benchmark results. Key steps include:
  1. Loading benchmark datasets and extracting states from the ESN reservoir using **Multi-State Pooling** (Mean, Max, Standard Deviation).
  2. Applying feature binarization and encoding techniques (**MinMax**, **Quantile Uniform**, and **Quantile Gaussian**).
  3. Training and evaluating the RAM-based **WiSARD** classifier.
  4. Measuring predictive metrics (Accuracy %, F1-Macro %) alongside hardware performance (**Fit Time**, **Inference Latency**, **Memory Usage KB**, and **FLOPs**) against the RidgeCV baseline.

* **`esn_grid_search.py`** (formerly `esnModelChecking.py`):
  The **Model Selection** script used to discover optimal ESN Reservoir configurations for each dataset.
  * Conducts an exhaustive Grid Search with **5-Fold Cross-Validation** over core reservoir hyperparameters: number of neurons (`units`), spectral radius (`sr`), and leaking rate (`lr`).
  * Outputs optimal hyperparameters to be used directly in `main_pipeline.py`.

* **`requirements.txt`**: 
  List of essential Python dependencies required for environment reproducibility.

---

## Installation & Setup

Follow these 3 steps to set up the environment on your local machine:

### 1. Clone this Repository
Clone this repository to your local machine:
```bash
git clone [https://github.com/your-username/esn-wisard-timeseries.git](https://github.com/your-username/esn-wisard-timeseries.git)
cd esn-wisard-timeseries
```

### 2. Create and Activate a Virtual Environment
It is recommended to use an isolated Python virtual environment:
```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
Install all required libraries with a single command:
```bash
pip install -r requirements.txt
```

> **Note on WiSARD Dependency:** Step 3 will automatically download, compile, and install all required packages—including the external **WiSARD C++ library** referenced via `git+` inside `requirements.txt`. No manual cloning or C++ compilation of the WiSARD repository is required.

---

## Running the Code

1. **(Optional) ESN Hyperparameter Tuning:**
   To find optimal Reservoir configurations for a dataset via 5-Fold Cross-Validation:
   ```bash
   python esn_grid_search.py
   ```

2. **Run the Main Pipeline:**
   To reproduce final accuracy benchmarks, binarization comparisons, and hardware latency/memory metrics:
   ```bash
   python main_pipeline.py
   ```
