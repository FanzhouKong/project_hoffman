"""specdenoise: neutral-loss plausibility denoising of MS/MS spectra.

Every fragment ion is judged by its neutral loss from the precursor: the loss must be the mass of a chemically
plausible formula (Seven Golden Rules, Kind & Fiehn 2007) that is a sub-formula of a plausible precursor ion
formula and leaves a plausible fragment ion. With the precursor formula known this is one bounded mass
decomposition per ion (tier T1); with only the precursor m/z the precursor formulas are inferred from the
accurate mass and ranked by the fragment intensity they explain (tier T0)."""
from .denoise import denoise, Result, Params      # noqa: F401

__version__ = "1.1.3"
