# Salary Prediction 

This repository contains a small salary prediction project with three main artifacts:

- `analysis_modeling.ipynb` — exploratory notebook that loads data, trains a model, and shows analysis.
- `salary_model.pkl` — trained regression model saved with `joblib` (used by the Streamlit app).
- `app.py` — Streamlit app that loads `salary_model.pkl` and exposes a simple UI for salary estimates.

**Contents**

- **Notebook** (`analysis_modeling.ipynb`)
  - Purpose: data cleaning, feature engineering, model training, evaluation and basic visualizations.
  - Key cells: data loading (pandas / openpyxl if reading .xlsx), preprocessing, model training (scikit-learn), saving model with `joblib.dump()`.
  - How to run: open the notebook in Jupyter or VS Code and re-run cells. If the kernel is the workspace venv, ensure dependencies are installed first.

- **Model** (`salary_model.pkl`)
  - Format: `joblib` serialized scikit-learn estimator (e.g., `LinearRegression` or similar).
  - Usage: loaded in `app.py` with `joblib.load('salary_model.pkl')` and used to call `.predict()` on input arrays of shape (n_samples, n_features).
  - Re-training: re-run the notebook training cells; after fitting `model`, call `joblib.dump(model, 'salary_model.pkl')` to overwrite the file.

- **App** (`app.py`)
  - Framework: Streamlit.
  - Behavior: reads user input (`Years of Experience`, `Job Rate`), constructs feature vector `X = [[years, jobrate]]`, loads `salary_model.pkl` and displays predicted salary.
  - Run locally (from project root):

```powershell
& "C:/Users/azerty/Documents/salary prediction/.venv/Scripts/python.exe" -m streamlit run "C:/Users/azerty/Documents/salary prediction/app.py"
```

  - Local Streamlit URLs are printed on start (e.g., `http://localhost:8503`).

Setup and Dependencies

- Recommended: use the provided virtual environment located at `.venv` in the project root (if present). The workspace Python executable is:

  `C:/Users/azerty/Documents/salary prediction/.venv/Scripts/python.exe`

- Minimal packages required (install into the venv):

```powershell
/salary prediction/.venv/Scripts/python.exe" -m pip install -U pip
/salary prediction/.venv/Scripts/python.exe" -m pip install -r requirements.txt
```

If you don't have a `requirements.txt`, you can install the minimal set used in the project:

```powershell
/salary prediction/.venv/Scripts/python.exe" -m pip install pandas scikit-learn joblib openpyxl streamlit
```

Notebook-specific tips

- If you see `ModuleNotFoundError: No module named 'openpyxl'` when reading `.xlsx` files, install `openpyxl` into the notebook kernel environment.
- In notebooks, prefer using `%pip install package` in a cell to ensure the kernel's environment receives the installed package.

App troubleshooting

- If `joblib` or `sklearn` is missing when starting the app, install them into the same interpreter used to run Streamlit (see commands above).
- Common Streamlit start command for PowerShell uses the `&` invocation to call the venv python.

Reproducible training snippet

Below is a minimal example to train and save a model (adapt to your dataset and preprocessing):

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# load data
df = pd.read_csv('your_data.csv')
# X should contain the same features the app expects, e.g., ['years', 'jobrate']
X = df[['years', 'jobrate']]
y = df['salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
joblib.dump(model, 'salary_model.pkl')
```

---

File references:

- Notebook: [analysis_modeling.ipynb](analysis_modeling.ipynb)
- App: [app.py](app.py)
- Model file (if present): [salary_model.pkl](salary_model.pkl)
