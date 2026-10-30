# PyInstaller spec for the standalone peak3d app:   pyinstaller --noconfirm packaging/peak3d.spec
# One folder per system (dist/peak3d/, or dist/peak3d.app on macOS) with Python, numba and peak3d inside.
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

import peak3d

ROOT = os.path.dirname(SPECPATH)
MAC, WIN = sys.platform == "darwin", os.name == "nt"

datas = [(os.path.join(ROOT, "peak3d", "web", "static"), os.path.join("peak3d", "web", "static"))]
datas += collect_data_files("psims") + collect_data_files("pyteomics")
hiddenimports = collect_submodules("peak3d", filter=lambda m: ".tests" not in m) + collect_submodules("psims")

a = Analysis(
    [os.path.join(SPECPATH, "peak3d_app.py")],
    pathex=[ROOT],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["peak3d.tests", "pytest", "IPython", "jupyter", "notebook", "PyQt5", "PyQt6", "PySide2", "PySide6"],
    # peak3d as .py source files: numba's on-disk cache needs the source of every cached function
    module_collection_mode={"peak3d": "py"},
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="peak3d",
    console=not (MAC or WIN),          # Linux: runs in a terminal; Windows/macOS: the small control window
    disable_windowed_traceback=False,
    upx=False,
    icon=os.path.join(SPECPATH, "mosa.ico") if WIN else None,     # packaging/make_icons.py
)
coll = COLLECT(exe, a.binaries, a.datas, name="peak3d", upx=False)
if MAC:
    app = BUNDLE(
        coll,
        name="peak3d.app",
        icon=os.path.join(SPECPATH, "mosa.icns"),
        bundle_identifier="edu.ucdavis.peak3d",
        info_plist={
            "CFBundleDisplayName": "peak3d",
            "CFBundleShortVersionString": peak3d.__version__,
            "CFBundleVersion": peak3d.__version__,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "14.0",      # the numpy/scipy wheels pip picks on the build runners
        },
    )
