# Quickstart

From the repository root:

```bash
pip install -r requirements.txt
python generate_data.py
python fitly_churn_analysis.py
python fitly_advanced_analysis.py
python fitly_utils.py
python EXAMPLES.py
```

Or open `notebooks/fitly_churn_analysis.ipynb` and run all cells. The first code cell selects the Matplotlib `Agg` backend and adds the repo root to `sys.path`.

`data/fitly_churn_analysis.xlsx` is created with seed 42 if it is not already there. Loaders resolve that path from the repo directory, not from the process working directory and not from a machine-specific upload path.

Expect 400 customers, 114 churned (28.5%), and 41 active Low or No Engagement customers. The random forest prints its own test accuracy (85.00% in the reference run of this repo). Four PNGs are written under `outputs/`.

To publish in DataCamp DataLab, upload the folder, install `requirements.txt`, run the notebook, then use DataLab's Publish / Share control while signed in to your own account.
