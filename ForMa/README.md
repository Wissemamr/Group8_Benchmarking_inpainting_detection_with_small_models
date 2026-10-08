# ForMa results visualization

Open `visualize_results_figs.ipynb` and select the Python environment used for
ForMa evaluation, then run the cells in order. It reads `results/results_excel.xlsx`
and shows macro metrics by category and mask type, overall averages, score
distributions, and manipulated-region size versus F1.

The sample section displays the manipulated image, ground-truth mask, actual
ForMa prediction, and a pixel error map. Change `SAMPLE_CATEGORY`,
`SAMPLE_SOURCE_ID`, and `SAMPLE_VARIANT` to inspect other samples. By default it
shows bbox and segm variant 0 for a source near the selected category's median
source-average F1. This selection is illustrative, not random.

Plotting requires `matplotlib`, `pandas`, `numpy`, `openpyxl`, and `ipykernel`.
Sample inference additionally requires the existing ForMa dependencies and the
same trusted checkpoint used for evaluation. The notebook verifies checkpoint
and input hashes, reuses the recorded thresholds, and reports whether regenerated
pixel counts match the workbook. CPU inference can take several minutes.

Figures are saved under `results/figures/<evaluation-id-prefix>/`; regenerated
binary masks are cached in `results/visualization_predictions/`. The workbook is
read-only. If it contains multiple configurations, set `EVALUATION_ID` explicitly.

For a terminal summary, run `python ForMa/show_results.py` from the project root.

## SD2-sp and profiling

Evaluate all downloaded SD2-sp categories from the project root:

```powershell
python ForMa/evaluate_tgif_category.py --subset sd2-sp --category all --device cpu
python ForMa/show_results.py --subset sd2-sp
```

Use `--check-only` to validate pairing without inference. Run timing summaries
are saved in `results/runs/<run-id>/run.json` and per-image timings in `images.csv`.
Use `--profile --limit 1 --output ForMa/results/sd2_profile.xlsx` for a small
Python call-profile diagnostic; profiling adds overhead and uses a separate
evaluation ID. Optional `psutil` enables sampled process-memory measurements.
See [forma.md, Section 15](forma.md#15-sd2-sp-evaluation-and-run-profiling-8-october-2026)
for measurement boundaries, examples and limitations.
