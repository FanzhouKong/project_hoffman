# Third-party software in the Mosa app

The Mosa app (the peak3d program, MIT licence) carries the following software. Each package's licence text is
in its metadata folder inside the app (`_internal/<package>-<version>.dist-info/licenses/` or `.../LICENSE`).

| Component | Licence | Home |
|---|---|---|
| asari (asari-metabolomics) | BSD 3-Clause | https://github.com/shuzhao-li-lab/asari |
| mass2chem, khipu-metabolomics, metDataModel | BSD | https://github.com/shuzhao-li-lab |
| jms-metabolite-services | Python Software Foundation licence | https://github.com/shuzhao-li-lab/JMS |
| pymzml | MIT | https://github.com/pymzml/pymzML |
| ms_entropy | Apache 2.0 | https://github.com/YuanyueLi/MSEntropy |
| json_tricks | BSD | https://github.com/mverleg/pyjson_tricks |
| pathvalidate | MIT | https://github.com/thombashi/pathvalidate |
| smiles-drawer 2.4.1 (structure drawings) | MIT | https://github.com/reymond-group/smilesDrawer |
| numpy, scipy, pandas, numba, llvmlite, matplotlib, statsmodels, lxml, pyteomics, psims | BSD-style / MIT / Apache 2.0 | their PyPI pages |
| Python | Python Software Foundation licence | https://www.python.org |

MassCube is not part of the app. Its licence (CC BY-NC 4.0) allows non-commercial use only; the app installs it
only when you ask for it and accept that licence, into a separate folder, with its own Python
(python-build-standalone, https://github.com/astral-sh/python-build-standalone).
