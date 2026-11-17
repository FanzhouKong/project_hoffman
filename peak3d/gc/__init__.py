"""GC-MS (electron ionisation) stages on top of the peak3d picker and alignment.

An EI source breaks every compound into tens of fragment ions that elute together, so the unit of a GC-MS result is
the compound and its spectrum, not the ion. The picker, the RT correction and the ion alignment are the LC ones and
are not changed; this subpackage adds what LC does not need:

  deconv    per file: the picked ions that share one apex and one peak shape become a component, and each component's
            spectrum is read from the raw centroids, an ion shared by two co-eluting compounds divided between them
  ri        retention index: finding an n-alkane or FAME ladder, and the RT -> RI map
  library   EI library (.msp, unit-mass or accurate-mass) and the spectrum search with a retention-index window
  pipeline  `peak3d gc OUTPUT`: runs the three on an existing peak3d output folder and writes OUTPUT/gc/

Nothing here is imported by the LC pipeline: an LC run gives the same files with or without this subpackage."""
