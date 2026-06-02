# GPU-Accelerated Samples

Jupyter notebooks demonstrating GPU acceleration using NVIDIA RAPIDS libraries on Microsoft Fabric and Azure VMs with NVIDIA GPUs.

> Based on examples from [NVIDIA RAPIDS](https://github.com/rapidsai) (Apache 2.0 License, Copyright NVIDIA Corporation). Modified and tested for Azure GPU VM environment.

## Requirements

- NVIDIA GPU (Tesla T4 or better)
- CUDA 12.x
- Python 3.12+
- RAPIDS 25.x (cuDF, cuML, cuCIM)
- Conda environment recommended

## Notebooks

| Notebook | Description |
|----------|-------------|
| [cudf-pandas-accelerator-demo.ipynb](cudf-pandas-accelerator-demo.ipynb) | Drop-in GPU acceleration for pandas with `cudf.pandas` — no code changes needed |
| [cudf-pandas-stock-analysis.ipynb](cudf-pandas-stock-analysis.ipynb) | Stock market data analysis using GPU-accelerated pandas (read, merge, resample, plot) |
| [cuml-scikit-learn-accelerator-demo.ipynb](cuml-scikit-learn-accelerator-demo.ipynb) | Drop-in GPU acceleration for scikit-learn (PCA, UMAP, KNN, HDBSCAN on activity recognition data) |
| [gpu-vs-cpu-compute-benchmark.ipynb](gpu-vs-cpu-compute-benchmark.ipynb) | Side-by-side GPU vs CPU benchmark for common compute operations |
| [rapids-dataframe-gpu-vs-cpu.ipynb](rapids-dataframe-gpu-vs-cpu.ipynb) | RAPIDS cuDF DataFrame operations compared to pandas on larger datasets |
| [multi-gpu-embedding-and-knn-search.ipynb](multi-gpu-embedding-and-knn-search.ipynb) | Multi-GPU text embedding generation and KNN similarity search using Ray + RAPIDS |

## cuCIM Medical Imaging

GPU-accelerated image processing for digital pathology and microscopy using [cuCIM](https://github.com/rapidsai/cucim).

> Based on examples from [rapidsai/cucim](https://github.com/rapidsai/cucim) (Apache 2.0 License, Copyright NVIDIA Corporation). Modified for Azure GPU VM environment.

| Notebook | Description |
|----------|-------------|
| [01-whole-slide-image-reading.ipynb](cucim_medical_imaging/01-whole-slide-image-reading.ipynb) | Reading and displaying multi-resolution whole-slide pathology images |
| [02-image-cache-performance.ipynb](cucim_medical_imaging/02-image-cache-performance.ipynb) | Tile caching strategies for faster repeated region access |
| [03-gabor-texture-classification.ipynb](cucim_medical_imaging/03-gabor-texture-classification.ipynb) | GPU vs CPU Gabor filter bank for texture classification |
| [04-random-walker-segmentation.ipynb](cucim_medical_imaging/04-random-walker-segmentation.ipynb) | GPU-accelerated random walker image segmentation |
| [05-vesselness-filter.ipynb](cucim_medical_imaging/05-vesselness-filter.ipynb) | GPU-accelerated vessel detection using Frangi/Sato filters |
