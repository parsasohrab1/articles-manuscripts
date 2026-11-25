# HVAC electricity outside opening hours (measured data)

Data: Zenodo records 23099351 and 23099489 (CC-BY-4.0, MD5 checked) and NASA POWER hourly temperature. Sources and hashes: `data/manifest.json`.

Run from this folder (Python 3.10; numpy, pandas, scikit-learn, matplotlib, python-docx):

    python -I code/hvac_study.py data results
    python -I code/hvac_paper.py data results paper

`code/download.py <dir> hvac` fetches the data again into `<dir>/hvac`.

Open points: opening hours are assumed; the town-hall time zone is ambiguous; no intervention was made, so no saving is claimed as realised; the simple "last week" baseline beat the trained models; references need checking; venue scope fit with Control Engineering Practice is weak.
