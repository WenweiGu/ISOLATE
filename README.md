# ISOLATE
![](https://img.shields.io/badge/version-1.0-blue.svg) 
![](https://img.shields.io/badge/language-python-orange.svg)

Here is the repository for the TOSEM paper "Identifying Performance Issues in Cloud Service Systems Based on Relational-Temporal Features." You can refer to the [DOI](https://dl.acm.org/doi/10.1145/3702978) for more details. 

## Overview

The following figure shows the overview of our framework ISOLATE, an automated method for detecting performance anomalies. ISOLATE consists of two main parts: the relational-temporal embedding part and the anomaly detection with the LC-VAE part. The relational-temporal embedding part captures relational and temporal patterns from the original metrics by employing graph attention to extract correlations among metrics and capturing temporal dependencies through GRU and temporal convolution. The anomaly detection part utilizes a label-conditional-VAE (LC-VAE) to distinguish anomalies from normal patterns. 

![ISOLATE](https://github.com/user-attachments/assets/bcab7c1b-d229-4eb6-ae20-a44af925e65f)

## Datasets

We only included one desensitized sample of our industrial dataset due to our company's confidentiality policy.

## Requirements

Python $\geq$ 3.8 is needed. Besides, the environment can be built by:
```pip install -r requirements.txt```

## Run the code
Cd to the current working directory and simply run:
```python train.py```

If you want to localize the anomalous metrics, please run:
```python train_rca.py```

### Citation
If you find our repo helpful, please cite our paper: 
```
@article{guidentifying,
  title={Identifying Performance Issues in Cloud Service Systems Based on Relational-Temporal Features},
  author={Gu, Wenwei and Liu, Jinyang and Chen, Zhuangbin and Zhang, Jianping and Su, Yuxin and Gu, Jiazhen and Feng, Cong and Yang, Zengyin and Yang, Yongqiang and Lyu, Michael R},
  journal={ACM Transactions on Software Engineering and Methodology},
  publisher={ACM New York, NY}
}
```
