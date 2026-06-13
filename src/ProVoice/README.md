# ProVoice: Driver State–Aware Adaptive Automation Assistant

## 1. Prerequisites

### Get the Source Code
- **Option 1: Download ZIP**
  - Go to the GitHub project page → Code → Download ZIP → unzip to a folder.
- **Option 2: Clone with Git**
  ```
  git clone <repository-link>
  ```

### Install an IDE
This project is written solely in Python. Recommended IDEs:
- Visual Studio Code (lightweight)
- PyCharm (full-featured Python IDE)

### Setup Environment
- Install Miniconda (or Miniforge, Anaconda, etc.): [Miniconda Quickstart](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions)
- Create and activate the Conda environment:
  ```
  cd D:/ProVoice   # replace with your local path
  conda env create -f environment.yml
  conda activate ProVoice
  pip install dash
  pip install dash-bootstrap-components
  ```

## 2. Install CARLA Simulator
- Download CARLA from the official site: [CARLA 0.10.0](https://carla.org/2024/12/19/release-0.10.0/)

## 3. Running the System
1. **Start Driving Simulation** (Open CARLA.exe first, then in a terminal):
   ```
   python drive.py
   ```
2. **Launch ProVoice**
   In a new terminal:
   ```
   python main.py participantid=001 environment=city secondary_task=none functionname="Adjust seat positioning" modeltype=combined state_model=xlstm w_fcd=0.7
   ```
   **Arguments:**
   - `participantid` – Participant ID
   - `environment` – Driving environment (`city` / `highway`)
   - `secondary_task` – Secondary task (`none` / `phone` / `drinking`)
   - `functionname` – Experimental function (e.g., "Adjust seat positioning")
   - `modeltype` – Decision model (`fcd` / `state` / `combined` )
   - `state_model` – Model used for state→LoA (`xgboost` / `xlstm`)
   - `w_fcd` – Weight for FCD in fusion (0–1)

## 4. Data Collection
We conducted a simulation-based user study to collect data for model training and initial evaluation. The experiment uses the CARLA simulator with a monitor, keyboard control, and a webcam to record the driver’s face. Participants (10–30 licensed drivers, balanced in age/gender) drive in various scenarios, with all monitoring systems running in real time. The refresh rate is ~20Hz for generated data (FCD, state features).

**Experimental Design:**
- Each scenario focuses on one driving function (14 total), with two factors varied: environment (`city`/`highway`) and secondary task (`none`/`phone`/`drinking`).
- Scenarios are distributed using a Latin square so each participant experiences a balanced subset.
- Each drive lasts ~50s: 10s Baseline (normal driving), 30s Task (main function + possible distraction), 10s Recovery (rest).
- After each scenario, participants report their preferred LoA (0–4). These subjective ratings help build the datasets.

**Data Annotation & Training:**
- Each 30s task segment is labeled with the participant’s LoA preference.
- Labels are assigned manually based on participant reports.
- Data is split into training/test sets for model development and evaluation.

**How to Run:**
Start a scenario with:
```
python main.py participantid=001 environment=city secondary_task=none functionname="Adjust seat positioning" modeltype=collection
```
Adjust arguments as needed for each participant and scenario.

## 5. Data Preprocessing
Data collected from the driving experiments is stored as raw JSONL logs. To prepare this data for model training and analysis, follow these four steps:

1. **Split into Segments**
   - The raw log contains continuous data from all sessions. Use the script below to split it into fixed-length segments (e.g., 600 samples per chunk (30 experiment seconds * 20 FPS)), each corresponding to a scenario or trial.
   ```
   python data/generate_id.py --in data/raw_data.jsonl --out data/with_segments.jsonl --chunk 600
   ```
   This creates `with_segments.jsonl`, where each entry is a segment with a unique ID.

2. **Generate Label File**
   - For each segment, generate a CSV file to annotate ground truth labels (LoA).
   ```
   python data/label_data.py --in data/with_segments.jsonl --out data/labels.csv
   ```
   The resulting `labels.csv` lists all Loa labels to be filled in.

3. **Manually Label**
   - Open `labels.csv` in Excel or another editor. For each segment, fill in the correct LoA label as reported by the participant after the scenario. This step ensures the model is trained on accurate, human-verified ground truth.

4. **Merge Labels into Dataset**
   - Combine the segment data and the annotated labels into a single JSONL file for model training and evaluation.
   ```
   python data/merge_label.py --in data/with_segments.jsonl --labels data/labels.csv --out data/labeled_data.jsonl
   ```
   The output `labeled_data.jsonl` contains all sensor data, conditions, and ground truth labels for each segment.

**Resulting files:**
- `with_segments.jsonl`: Segmented data with unique IDs
- `labels.csv`: Annotation file for experimental conditions and LoA labels
- `labeled_data.jsonl`: Final dataset for model training and evaluation

This process ensures that each data segment is accurately labeled and ready for downstream machine learning tasks.

## 6. Model Training
- Train State→LoA (xLSTM)
  - This uses the official [`nx-ai/xlstm`](https://github.com/NX-AI/xlstm)
    package (`xlstm==2.0.5`), which is now a normal `uv` dependency — **no
    manual repo checkout or separate conda env is needed**. It runs on the
    CPU-compatible mLSTM `xLSTMBlockStack` path.
  - The classifier is **single-label, 5-class** (LoA 0–4).
  - **A real labeled dataset is required to retrain.** No trained checkpoint
    is committed to this repo. Run the data pipeline first
    (`data/generate_id.py` → `data/label_data.py` → `data/merge_label.py`,
    see section 5); `label_data.py` produces a **blank label template** that
    the researcher must fill in by hand before merging.
  ```
  python -m ProVoice.train_XLSTM --in data/labeled_data.jsonl --out trained_models/state_xlstm.pt --epochs 30 # hyperparameter tuning is needed based on your dataset
  ```

## 7. Run Decision Engines
> **Note:** xLSTM inference runs on CPU. If `trained_models/state_xlstm.pt`
> is absent, the engine falls back to FCD / LoA 0.

- **FCD→LoA (XGBoost):**
  ```
  python main.py ... modeltype=fcd
  ```
- **State→LoA (xLSTM):**
  ```
  python main.py ... modeltype=state state_model=xlstm
  ```
- **Combined Fusion (FCD + xLSTM):**
  ```
  python main.py ... modeltype=combined state_model=xlstm w_fcd=0.7
  ```
- **Evaluation:**
  ```
  python eval.py --in data/labeled_data.jsonl --outdir reports/eval --title "ProVoice LoA Evaluation"
  ```

## 8. Dashboard
When running `main.py`, a browser window will open automatically. The dashboard displays:
- Driver video (mocked/simulated)
- Secondary task detection (phone, drinking, smoking)
- Physiological signals (mocked HR, HRV)
- LoA predictions in real time
- Decision engine logs

![ProVoice Driver State Dashboard](image/dashboard.png)

---
