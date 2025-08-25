This repository contains the official implementation of the paper:

**Computationally Efficient and Generalizable Machine Learning Algorithms for Seizure Detection from EEG Signals**.

Automated seizure detection from scalp EEG is critical for epilepsy management, yet existing algorithms often struggle to balance computational efficiency, predictive performance, and generalizability across diverse patient populations.

This study investigates the **ROCKET framework** and complementary state-of-the-art frameworks in time series classification and seizure detection tasks, including:
- **Detach-ROCKET**
- **Detach Ensemble**
- **STFT-based feature transform**
- **catch22**
- **EEGNet**

Models were trained on the **TUSZ dataset** and evaluated on **TUSZ** and the independent **Siena dataset** to assess inter-subject and cross-dataset generalizability.

## Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/zheyun-shou/eeg_seizure.git

conda create -n rapids-25.04 -c rapidsai -c conda-forge -c nvidia  \
    rapids=25.04 python=3.10 'cuda-version>=12.0,<=12.8'
source activate rapids-25.04
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -c pytorch

pip install pybids mne==1.9.0 tabulate sktime numpy==1.24.3 scikit-learn==1.5.2 seaborn
python -m pip install -U timescoring

cd eeg_seizure
```

When using EEGNet, it is recommended to create a new conda environment, then follow the [instructions](https://github.com/aliasvishnu/EEGNet).

When using catch22, follow the installing [instructions](https://github.com/DynamicsAndNeuralSystems/catch22).

## Usage

### **1. Data Preparation**
- Download the datasets:
  - [TUSZ](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/#c_tusz)
  - [Siena Scalp EEG Database](https://physionet.org/content/siena-scalp-eeg/1.0.0/)
- Convert the data to BIDS, we use https://github.com/esl-epfl/epilepsy2bids.

### **2. Training a Model**

Edit config.yaml to adjust the setting.
Then run:
```bash
python pipeline.py
```

### **3. Evaluating a Model**

Edit the model name in evaluate.py, choose the dataset. Then run:
```bash
python evaluate.py
```

## References & Related Repositories
This work builds on and extends the following projects:
- Detach-ROCKET: [publication](https://link.springer.com/article/10.1007/s10618-024-01062-7), [repo](https://github.com/gon-uri/detach_rocket).
- Detach Ensemble:[publication](https://dl.acm.org/doi/10.1007/978-3-031-77066-1_6), [repo](https://github.com/gon-uri/detach_rocket).
- STFT-based feature transform: [publication](https://ieeexplore.ieee.org/document/4801967).
- catch22: [publication](https://link.springer.com/article/10.1007/s10618-019-00647-x), [repo](https://github.com/DynamicsAndNeuralSystems/catch22).
- EEGNet: [publication](https://arxiv.org/abs/1611.08024), [repo](https://github.com/aliasvishnu/EEGNet).
- [the Seizure detection challenge (2025)](https://epilepsybenchmarks.com/challenge-description/) and [timescoring](https://github.com/esl-epfl/timescoring) framework provided by the challenge.


