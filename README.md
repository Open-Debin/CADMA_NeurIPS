# Minimal Gaussian Experiments for Fig. 3 Row 1

This directory contains the minimal code needed to reproduce the Gaussian
mode-dropping experiments shown in Fig. 3, row 1 of the NeurIPS submission.
For more context, see Sections 4.1 and 4.2 in the main paper.

The script covers both settings:

- `density_shift`: dropped mass is redistributed to the dominant in-domain mode
- `outlier_pollution`: dropped mass is replaced by samples from a separate OOD Gaussian

To keep the artifact lightweight for review and debugging, the default run uses
`100` samples per Gaussian mode. The paper-scale run can be obtained with
`--n-per-mode 3571`, matching the notebook setting.

## What Is Included

The experiment uses a 7-mode Gaussian mixture and reports only the metrics used
in this simplified artifact:

- `Recall`
- `Coverage`
- `ClippedCoverage`
- `Coverage_our`
- `R_cap`
- `D_drop`
- `D_collapse`

The plotting code uses the same colors, markers, line styles, and separate
legend format as the paper figures. The core plot also includes the black dotted
reference line used in the notebook.

## Files

- `run_gaussian_experiments.py`: main entry point
- `metrics/`: local metric implementations used by the script
- `requirements.txt`: Python package dependencies

The script is self-contained for the Gaussian experiments: it does not depend on
the original notebooks or the external TopPR data generator.

## Install

This code requires Python `>=3.9` and was tested with Python `3.10.0`.

Using pip:

```bash
pip install -r requirements.txt
```

Using conda:

```bash
conda env create -f environment.yml
conda activate gaussian-mode-dropping
```

## Simple Test

Run both Gaussian experiments with 100 samples per mode:

```bash
python run_gaussian_experiments.py
```

The default output directory is:

```text
outputs/gaussian_mode_dropping
```

## Paper-Scale Run

For the paper-scale Gaussian experiment:

```bash
python run_gaussian_experiments.py --n-per-mode 3571
```

## Outputs

For each setting, the script writes:

- `*.csv`: metric values at each dropping/outlier ratio
- `*_baselines_plus_cadma_core_PLOT.png`: Recall/Coverage/ClippedCoverage plus `Coverage_our` and `R_cap`
- `*_baselines_plus_cadma_core_LEGEND.png`: separate legend for the core plot
- `*_cadma_decomposition_PLOT.png`: `D_drop` and `D_collapse`
- `*_cadma_decomposition_LEGEND.png`: separate legend for the decomposition plot

The output names include the setting, number of samples per mode, and CADMA
`k`, for example:

```text
gaussian_density_shift_n100_cadmak30.csv
gaussian_outlier_pollution_n100_cadmak30.csv
```
