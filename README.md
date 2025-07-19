# FedFLI
This is a Pytorch implementation of FedFLI: Federated Learning for Edge-assisted Traffic Flow Prediction with LLM-enabled Distillation and Variational Information Bottleneck.

## Requirement
* Python 3.11
* torch == 2.1.0
* cuda == 12.1

## Data
We provide the processed NYC-Taxi and CHI-Bike datasets，the original data please refer to the repository of [ST-LLM](https://github.com/ChenxiLiu-HNU/ST-LLM).

## Train and Test
Step 1: Process the teacher model:

```python
python run_LM4ST.py
```

Step 2: Train and test the student model:

* For MLP student model
```python
python run_MLP.py
```

* For STGCN student model
```python
python run_STGCN.py
```

The above commands are run only according to the default parameter settings. All parameters can be modified according to your needs.
