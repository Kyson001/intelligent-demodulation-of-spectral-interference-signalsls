# Visual and source QA

2026-11-16: all four final PNGs inspected panel by panel after final export.

- Font audit: minimum 7 pt in figures 01/02, 5.6 pt in 03/04 including log exponents;
  zero glyph runs below 5 pt.
- Collision audit: zero FAIL and zero WARN for all four PDFs. Contained fill
  overlays are internal log tick glyph boxes and accepted after visual review.
- Source validator: no FAIL. Four advisories reviewed: PNG is a report preview,
  not TIFF submission; 300 dpi is intentional; report-width figure is intentional;
  every log10 input is explicitly floored with np.maximum and ECDF log axes contain
  strictly positive values (A zero differences are excluded by definition, not observations).
- Representative plots use first/middle/last indices fixed before comparison.
- All 4000 records contribute to population ECDFs, including duplicate records;
  no independence or significance inference made.
- Curves are normalized as axis labels state; relative FFT dB is not SNR.
- Known-truth errors occur only in clearly labelled synthetic controls.
- Uncertainty marks are empirical P95, not confidence intervals.
- No decorative AI image generation, raster substitutions, or omitted failure cases.
