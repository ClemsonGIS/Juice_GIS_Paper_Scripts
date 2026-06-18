# Juice_GIS_Paper_Scripts

Scripts and benchmarking harness for the PEARC short paper **"Extending HPC GPUs to Windows-Based GIS Users via Juice: A Model for Broadening Access to Campus Supercomputing."**

These scripts benchmark deep learning training in ArcGIS Pro on two setups: a **local GPU** and a **remote A100 accessed over the network via [Juice](https://www.juicelabs.co/) (GPU-over-IP)**. The workload is an RT-DETR v2 object-detection model trained at batch sizes 16, 32, and 64.

Because Juice works transparently at the driver level, ArcGIS Pro is unaware the GPU is remote. The training script is therefore identical for both setups — only the runtime environment and GPU-sampling command differ.

## Files

| File | Role |
|------|------|
| `PythonTest_ModelBuilder.py` | ModelBuilder export that runs the three training jobs (local run). |
| `PythonTest_ModelBuilder_Juice.py` | Same export, used for the Juice run (intentionally identical). |
| `train_mb_monkey_wrapper.py` | Logs time and GPU usage per training call; samples GPU via local `nvidia-smi`. |
| `train_mb_monkey_wrapper_juice.py` | Same harness, but requires `juice run nvidia-smi`. |

The wrappers time each `TrainDeepLearningModel` call and write per-run GPU logs plus a `results_master.csv` (wall-clock time, GPU utilization, GPU memory).

## Requirements

- Windows with **ArcGIS Pro** `[VERSION 3.6]`, Image Analyst extension, and Deep Learning Libraries installed.
- Run with ArcGIS Pro's bundled Python (`arcgispro-py3`).
- For the Juice run: the **Juice client** installed and on `PATH`, connected to a remote GPU host.

## Usage

Run from an ArcGIS Pro Python prompt, passing the ModelBuilder script as the argument.

```bat
:: Local
python train_mb_monkey_wrapper.py PythonTest_ModelBuilder.py

:: Juice
python train_mb_monkey_wrapper_juice.py PythonTest_ModelBuilder_Juice.py
```

## Reproducing Results

Clone the repository, prepare ArcGIS Pro training samples with **Export Training
Data For Deep Learning**, and set the paths below for your local project. Run the
local wrapper for a directly attached GPU and the Juice wrapper for the remote GPU
configuration. Each wrapper creates a timestamped results folder containing
`results_master.csv`, `gpu_log_*.csv`, and `gpu_proc_*.log`.

## Configuration

Configure paths with environment variables or by editing the variables near the
top of each script:

- `PROJECT_ROOT` — base folder used for default relative paths.
- `TRAINING_SAMPLES_DIR` — ArcGIS deep learning training-data export folder.
- `WORKSPACE_GDB` — ArcGIS workspace/scratch geodatabase.
- `MODEL_OUTPUT_ROOT` — where trained model outputs are written.
- `RESULTS_ROOT` — where wrapper benchmark logs are written.
- `IMAGE_ANALYST_TOOLBOX` — optional override for the ArcGIS Image Analyst toolbox.

Training data is not included. Use **Export Training Data For Deep Learning** in ArcGIS Pro to create your own samples and point the scripts at them.

## Notes

The local and remote setups use different GPUs, so these scripts measure the end-to-end experience of each configuration rather than isolating Juice's network overhead. Metrics are single-run. See the paper for details.

## Citation

`[FULL CITATION — authors, title, PEARC year, DOI]`
