# Day-ahead gas distribution offtake and balancing margin (measured data)

Data: ENTSOG Transparency Platform (daily physical exit flow, 2022-2025) and NASA POWER daily temperature. Sources and hashes: `data/manifest.json`. Check the terms of both sources before redistributing the files.

Run from this folder (Python 3.10; numpy, pandas, scikit-learn, matplotlib, python-docx):

    python -I code/gas_study.py data results
    python -I code/gas_paper.py results paper

`code/download.py <dir> gas` fetches the data again into `<dir>/gas`.

Open points: same-day observed temperature stands in for a weather forecast; the best model was chosen on test results; margin coverage is below 95 %; check overlap with the EIGN 2027 conference paper on the same product; venue template and word limit unchecked; references need checking.
