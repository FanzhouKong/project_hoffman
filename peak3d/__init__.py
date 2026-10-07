"""peak3d: 3D apex peak picking on the MS1 centroid point cloud, plus reference-free RT correction.

A feature is a basin of the (rt, m/z, intensity) surface: every centroid climbs to its
highest neighbour in adjacent scans and in m/z, basins whose apex is not persistent are merged
into their neighbour, and the surviving basins are described, gated and scored. Runs of a study
are then aligned on their own most credible features (no internal standards) and grouped into
one table.

Nothing here imports from ``bench/`` or reads ground truth; the package is self-contained.
"""
__version__ = "0.1.0"
