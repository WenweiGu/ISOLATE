# ISOLATE
![](https://img.shields.io/badge/version-1.0-blue.svg) 
![](https://img.shields.io/badge/language-python-orange.svg)

Here is the repository for the TOSEM submission "Correlation-Aware Performance Issue Identification in Cloud Service Systems". 

## Overview

The following figure shows the overview of our framework ISOLATE, an automated method for detecting performance anomalies. ISOLATE consists of two main parts: the relational-temporal embedding part and the anomaly detection with LC-VAE part. The relational-temporal embedding part captures relational and temporal patterns from the original metrics by employing graph attention to extract correlations among metrics and capturing temporal dependencies through GRU and temporal convolution. The anomaly detection part utilizes a label-conditional-VAE (LC-VAE) to distinguish anomalies from normal patterns. 

![RTAnomaly](https://github.com/ICSE24-Submission/RTAnomaly/assets/131580646/5c4b24c6-d371-4518-8c92-f7c812d8ae19)

## Datasets

We only put one sample of our industrial dataset and SMD dataset here (due to the limit of space). However, upon acceptance of this work, we will release all datasets used in our study, including raw data and all labels, and any other relevant materials necessary for replication and analysis of our findings.

## Requirements

Python $\geq$ 3.8 is needed. Besides, the environment can be built by:
```pip install -r requirements.txt```

## Run the code
Cd to the current working directory and simply run:
```python train.py```

If you want to localize the anomalous metrics, please run:
```python train_rca.py```
