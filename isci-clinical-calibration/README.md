# Post-hoc probability calibration on small clinical tabular data: a repeated cross-validation study of four calibrators, five classifiers and fifteen public data sets

Target venue: Information Sciences (Elsevier). Draft for author review; not submitted.

Data: 15 UCI Machine Learning Repository archives; URLs, sizes and SHA-256 hashes in `data/manifest.json`.

Run from this folder (Python 3.10; numpy 2.1, pandas 2.2, scikit-learn 1.5, scipy 1.14, joblib, matplotlib, python-docx, xlrd):

    python -I code/download.py data
    python -I code/study.py data results
    python -I code/analyse.py results
    python -I code/diagnostics.py data results
    python -I code/isci_paper.py results .

Open points before submission: authors, affiliations, funding and declarations; verify every reference and the UCI licences; no sensitivity analysis for duplicate removal; default hyper-parameters only. The study proposes no new method, so its fit and novelty for Information Sciences are weak and rejection at screening is a real risk. The ClinCal tracking row on the dashboard says an older submission could not be verified: confirm that no calibration paper on these data is already under review. ReadmitCal (diabetes readmission calibration) is a separate, already submitted paper on a different data set.
