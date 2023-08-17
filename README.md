# Predictive Maintenance Experiments

Scripts to experiment with predictive maintenance & related methods.

## Scripts index

- [Milling machine failure classification - XGBoost](./milling_xgboost.py)
  <details>
  <summary>Results</summary>

      Raw results - no class imbalance handling

      Class counts: Counter({0: 9652, 5: 112, 1: 95, 3: 78, 2: 45, 4: 18})
      train[raw]:  (7000, 6) (7000,)
      test[raw]:  (3000, 6) (3000,)

      XGBoost[raw] scores:
                            precision   recall  f1-score  support
              No Failure       0.99      1.00      0.99      2896
           Power Failure       0.68      0.75      0.71        28
       Tool Wear Failure       0.00      0.00      0.00        14
      Overstrain Failure       0.86      0.78      0.82        23
         Random Failures       0.00      0.00      0.00         5
      Heat Dissipation Failure 1.00      1.00      1.00        34

                accuracy                           0.99      3000
               macro avg       0.59      0.59      0.59      3000
            weighted avg       0.98      0.99      0.98      3000


      Downsampling results

      Class counts: Counter({0: 18, 1: 18, 2: 18, 3: 18, 4: 18, 5: 18})
      train[downsampling]:  (75, 6) (75,)
      test[downsampling]:  (33, 6) (33,)

      XGBoost[downsampling] scores:
                          precision     recall   f1-score   support
              No Failure       0.43      0.50      0.46         6
           Power Failure       0.40      0.40      0.40         5
       Tool Wear Failure       1.00      1.00      1.00         5
      Overstrain Failure       0.67      0.67      0.67         6
         Random Failures       0.50      0.67      0.57         6
      Heat Dissipation Failure 1.00      0.40      0.57         5

                accuracy                           0.61        33
               macro avg       0.67      0.61      0.61        33
            weighted avg       0.65      0.61      0.61        33

      SMOTE results

      Class counts: Counter({0: 9652, 1: 9652, 2: 9652, 3: 9652, 4: 9652, 5: 9652})
      train[SMOTE]:  (40538, 6) (40538,)
      test[SMOTE]:  (17374, 6) (17374,)

      XGBoost[SMOTE] scores:
                          precision    recall   f1-score  support
              No Failure       1.00      0.96      0.98      2896
           Power Failure       1.00      1.00      1.00      2896
       Tool Wear Failure       0.98      1.00      0.99      2895
      Overstrain Failure       1.00      1.00      1.00      2896
         Random Failures       0.99      1.00      0.99      2896
      Heat Dissipation Failure 1.00      1.00      1.00      2895

                accuracy                           0.99     17374
               macro avg       0.99      0.99      0.99     17374
            weighted avg       0.99      0.99      0.99     17374

  </details>

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