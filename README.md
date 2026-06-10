# VAE Credit Card Fraud Detection

This project implements an unsupervised credit card fraud detection model using a Variational Autoencoder (VAE) in PyTorch. The model is trained only on normal transactions and identifies fraud by assigning high anomaly scores to transactions that are poorly reconstructed by the VAE.

The notebook uses the widely used Kaggle credit card fraud dataset, where transactions are highly imbalanced and the original features have been anonymized using PCA.

## Project Overview

Fraud detection is naturally suited to anomaly detection because fraudulent transactions are rare and often differ from the dominant pattern of legitimate behavior. Instead of training a supervised classifier on a small number of fraud labels, this project trains a VAE only on normal transactions. The model learns the structure of legitimate transactions and flags observations that deviate from this learned distribution.

The anomaly score is computed as:

[
\text{Anomaly Score} = \text{Reconstruction Loss} + \text{KL Divergence}
]

Transactions with scores above a chosen threshold are classified as fraud.

## Dataset

The project uses the Kaggle Credit Card Fraud Detection dataset:

* `Time`: seconds elapsed between each transaction and the first transaction
* `Amount`: transaction amount
* `V1`–`V28`: PCA-transformed anonymized transaction features
* `Class`: target variable

  * `0`: normal transaction
  * `1`: fraudulent transaction

The dataset is highly imbalanced, with only 492 fraud cases among more than 280,000 transactions.

## Methodology

### 1. Data Preprocessing

The notebook separates normal and fraudulent transactions before training.

Only normal transactions are used for training, validation, and normal test evaluation:

* 70% of normal transactions for training
* 15% of normal transactions for validation
* 15% of normal transactions for testing
* All fraud transactions are held out for final evaluation

The `Time` and `Amount` variables are standardized using statistics computed only from normal transactions. This avoids leakage from fraudulent observations during preprocessing.

### 2. Model Architecture

The model is a beta-VAE implemented in PyTorch.

Encoder:

```text
Input dimension: 30
Linear(30, 64) + ReLU
Linear(64, 32) + ReLU
Latent mean: Linear(32, 8)
Latent log-variance: Linear(32, 8)
```

Decoder:

```text
Linear(8, 32) + ReLU
Linear(32, 64) + ReLU
Linear(64, 30)
```

The latent dimension is set to 8, forcing the model to learn a compressed representation of normal transaction behavior.

### 3. Training Objective

The VAE is trained by minimizing the evidence lower bound loss:

```text
Loss = Reconstruction Loss + beta * KL Divergence
```

where `beta = 4`.

The reconstruction loss encourages the decoder to reproduce normal transactions accurately, while the KL term regularizes the latent representation.

### 4. Early Stopping

Training is performed with Adam using a learning rate of `1e-3`.

The notebook uses early stopping based on validation loss:

* Maximum epochs: 200
* Patience: 10 epochs
* Best model saved as `best_vae.pt`

### 5. Anomaly Scoring

After training, the model computes an anomaly score for each transaction:

```text
Anomaly Score = Reconstruction Error + KL Divergence
```

Normal transactions should receive lower scores because the model was trained to reconstruct them. Fraudulent transactions should receive higher scores because they deviate from the learned normal pattern.

### 6. Threshold Selection

The classification threshold is selected using only validation-normal scores.

The final threshold is the 99th percentile of the validation-normal anomaly scores. This choice keeps the false-positive rate relatively low while still detecting a large fraction of fraud cases.

The notebook also compares thresholds at the 90th, 95th, 97th, and 99th percentiles to show the precision-recall tradeoff.

## Results

Final operating threshold:

```text
99th percentile of validation-normal anomaly scores
```

Main results:

* Fraud recall: approximately 80%
* Fraud detected: 394 out of 492 cases
* False alarms: 569 normal transactions
* False-positive rate: approximately 1.3%
* Precision: approximately 46%
* AUPRC: 0.6735

Given the severe class imbalance, the AUPRC is substantially better than the random baseline of roughly 0.0017.

## Interpretation

The model successfully learns the structure of normal transactions and identifies a large fraction of fraud cases without using fraud labels during training.

However, the model should not be interpreted as producing calibrated fraud probabilities. The anomaly score is a relative measure of abnormality, not a probability. A transaction with a high score is unusual under the learned normal-transaction representation, but it is not necessarily fraudulent.

In practice, this kind of model would be most useful as a screening tool that sends high-risk transactions to a downstream review system or human analyst.

## Strengths

* Fully unsupervised training setup
* Trained only on normal transactions
* Avoids relying on the small number of fraud labels
* Uses validation-normal scores for threshold selection
* Maintains a relatively low false-positive rate
* Achieves strong fraud recall despite extreme class imbalance

## Limitations

* Precision is moderate, so many alerts would still be false positives
* PCA-transformed features make interpretation difficult
* The anomaly score is not a calibrated probability
* The model does not use transaction metadata beyond the anonymized features
* Performance depends strongly on the chosen threshold

## Repository Contents

```text
.
├── 1_VAE_Fraud_Detection_improved.ipynb   # Main notebook
├── best_vae.pt                            # Saved model checkpoint, generated after training
└── README.md                              # Project documentation
```

## Requirements

The notebook uses the following Python libraries:

```text
pandas
numpy
torch
scikit-learn
matplotlib
```

You can install the dependencies with:

```bash
pip install pandas numpy torch scikit-learn matplotlib
```

## How to Run

1. Download the Kaggle Credit Card Fraud Detection dataset.
2. Place `creditcard.csv` in your working directory.
3. Update the dataset path in the notebook if needed:

```python
df = pd.read_csv("creditcard.csv")
```

4. Run the notebook from top to bottom.
5. The best model checkpoint will be saved as:

```text
best_vae.pt
```

## Possible Improvements

Several extensions could improve the project:

* Compare the VAE against classical anomaly detection methods such as Isolation Forest, One-Class SVM, and Local Outlier Factor.
* Tune the latent dimension, beta value, batch size, and threshold percentile.
* Use a validation set with a small number of fraud labels for threshold calibration.
* Add ROC-AUC, confusion matrices, and cost-sensitive evaluation.
* Calibrate anomaly scores into approximate probabilities.
* Test denoising VAEs or more expressive architectures.
* Use a supervised baseline such as logistic regression, random forest, or gradient boosting for comparison.

## Conclusion

This project demonstrates how a Variational Autoencoder can be used for unsupervised fraud detection under severe class imbalance. By learning only from normal transactions, the model identifies fraudulent transactions as high-anomaly observations. The final model detects approximately 80% of fraud cases while keeping the false-positive rate near 1.3%, making it a reasonable prototype for anomaly-based fraud screening.
