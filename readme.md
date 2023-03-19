# Hierarchical Transformer for Brain Computer Interface

**Update (2.8.2022)**: Initial code release

**Update (9.8.2022)**: Update `generate_dataset.py` in handling dataset code and method definition.

**Update (29.8.2022)**: Update `main.py` in handling two evaluation paradigms.

**Update (05.9.2022)**: Update `generate_dataset.py` adding Channel Normalization

## Introduction
TBA

## Requirement
### Environment
Make sure you have `Python==3.9` installed on the computer.

### Installation
1. [PyTorch](pytorch.org)
```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch
```

2. [MOABB](http://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014001.html)
```bash
pip install moabb==0.4.5
```
   Please note that this library will download the dataset from third party website.

3. [SciPy](scikit-learn.org)
```bash
pip install scikit-learn==1.8.1
```

4. [Einops](https://pypi.org/project/einops/)
```bash
pip install einops==0.4.1
```

3. [tqdm](https://pypi.org/project/tqdm/)
```bash
pip install tqdm
```

## Usage
Please follow these following steps to run the code.
### Download Dataset
Open [`generate_dataset.py`](https://github.com/skepsl/BCITransformer/blob/main/generate_dataset.py) code through the IDE.
This code aims to download and generate the corresponding MI dataset for each subject. First, it will download raw datasets from MOABB and save it in the local directory. We suggest that the computer has at least 5GB free capacity to store all original and preprocessed datasets.

The argument for `dataset` is either `BCIC`, `PhysioNet`, `Cho`, `Lee`.

Example to generate **Dataset I**, use:
```bash
Dataset(dataset='BCIC').get_dataset()
```

### Training and Evaluation
The code to train and evaluate this paradigm is inside [`main.py`](https://github.com/skepsl/BCITransformer/blob/main/main.py). 
The argument for `dataset` must be either `BCIC`, `PhysioNet`, `Cho`, or `Lee`. The fold must be an integer number between 1-10. The subject must be an integer represent the subject ID. 

Example to  train with 10F-CV for **Dataset I**, **subject 1**, **fold 1**, use:
```bash
Train(dataset='Lee').SVtrain(subject=1, fold=1) 
```

Example to  train with LOSO-CV for **Dataset I**, **subject 1**, use:
```bash
Train(dataset='Lee').SItrain(subject=1)
```

## Citation

```
@article{tba2023,
  title={BCI Transformer},
  author={TBA},
  journal={TBA},
  year={TBA}
}
```


