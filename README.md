# Predictive Maintenance Experiments

Scripts to experiment with predictive maintenance & related methods.

## Scripts index

- [Milling machine failure classification - XGBoost & class imbalance handling](./experiments/failure_classification/milling_machines_class_balancing.py)
- [Milling machine failure classification - Tree methods](./experiments/failure_classification/milling_machines_classifier_comparison.py)

## Setup

```bash
# Pre-requirements: Python 3.11 & pip 22.3.1

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Resources

- Predictive Maintenance: Predicting Machine Failure using Sensor Data with XGBoost and
  Python. By Florian
  Follonier [@flo7up](https://github.com/flo7up). ([Article](https://www.relataly.com/predictive-maintenance-predicting-machine-failure-with-python/10618/) | [Code](https://github.com/flo7up/relataly-public-python-tutorials/blob/master/02%20Classification/022%20Predicting%20Machine%20Malfunction%20of%20Milling%20Machines%20in%20Python.ipynb) | [Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset))
- Explainable Artificial Intelligence for Predictive Maintenance Applications. By Stephan
  Matzka. ([Paper](https://github.com/linomp/pdm-experiments/files/12337616/matzka2020.pdf))
- Handling Imbalanced Datasets in ML - codebasics. ([Video](https://www.youtube.com/watch?v=JnlM4yLFNuo))
- SMOTE & ADASYN in
  sklearn. ([Docs](https://imbalanced-learn.org/stable/over_sampling.html#from-random-over-sampling-to-smote-and-adasyn))

### Datasets

- [CNC MIll Tool Wear](https://www.kaggle.com/datasets/shasun/tool-wear-detection-in-cnc-mill)
- [Tool wear dataset of NUAA_Ideahouse](https://ieee-dataport.org/open-access/tool-wear-dataset-nuaaideahouse)
- [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)