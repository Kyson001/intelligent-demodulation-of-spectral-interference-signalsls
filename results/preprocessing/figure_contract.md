# Figure and experiment contract

Date: 2026-11-16. Python/matplotlib; original quantitative grids, no external template.

Question: does envelope correction or Hann apodization improve extraction without
distorting OPD/phase or suppressing information needed by a neural network?

Design frozen before population comparison: A=uniform k + scalar DC removal;
B=slow-background subtraction / guarded background proxy; C=B times periodic Hann.
These are paired observations, not independently trained models. Fixed phase support,
weights and sideband width across all three methods. Config is committed as JSON.

01: representative median-index input -> raw and envelope -> local spectrum ->
FFT main lobe -> full phase -> detrended phase. Describe coordinate/normalization.
02: all 4000 paired spectra, show spectral peak width, phase fit residual, phase
changes and OPD changes. These are consistency metrics, not independent accuracy.
03: synthetic known-OPD/phase signals; clean/noisy/offset conditions and seeds.
Primary evidence is true phase and OPD error; normalization/edge-noise comparison
explains the limitation of envelope compensation for network inputs.

No observations excluded because they weaken a method. Failed extraction is counted.
Duplicate spectra retained in descriptive population; do not infer independent n.
Real data labels and intensity unit unverified. Synthetic results are a numerical
control, not validation on a new instrument or proof of neural-network performance.

Representative indices: first, middle and last (1,2000,4000); no best-case selection.
Export PNG, editable SVG and PDF. Text >=7pt (>=5pt rendered). Source data: metrics CSV,
trace NPZ and configuration. White background, blue/orange/purple method palette.
Before delivery: source validator, PDF font audit, collision audit and visual review.
