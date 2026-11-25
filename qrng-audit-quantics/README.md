# Position-dependent bias and randomness extraction in public IBM QRNG sequences (QUANTICS 2027)

Data: Zenodo record 10542216 (CC-BY-4.0), not stored here. See `data/manifest.json`; extract the zip into `data/raw`.

Run from this folder (Python 3.10; numpy, pandas, scipy, matplotlib, python-docx):

    python -I code/qrng_study.py data/raw/S_2023_trials code results
    python -I code/qrng_paper.py data/raw/S_2023_trials results paper

`code/qrngaudit` is copied from branch `research/real-data-ibm` of the product repository articles-qrng-audit.

Open points: the position-stream extractor and the screen threshold were chosen after seeing the data and need an independent run; only five basic tests and two SP 800-90B estimators; circuit definitions are not in the archive; the main non-randomness finding was already reported by Root and Becker (arXiv 2401.12250); authors, declarations, references and venue length limit are unchecked. The abstract stored on the dashboard and the older draft in the product repository use unscreened numbers and are superseded.
