# IGD and Cyberbullying Detection: A Deep Learning Approach

This repository contains the source code and datasets used for the research paper titled **"A Novel Machine Learning & Deep Learning Approach Based Dataset Efficacy Study in Predicting Mental Health Outcomes from Internet Gaming Disorder and Cyberbullying"**.
[https://doi.org/10.6084/m9.figshare.27266961]

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Datasets](#datasets)
- [Installation](#installation)
- [Running the Code](#running-the-code)
- [Expected Results](#expected-results)

## Overview

This repository provides the code for predicting mental health outcomes associated with Internet Gaming Disorder (IGD) and Cyberbullying using machine learning and deep learning models. Models like Logistic Regression, Random Forest, Ensemble Models, CNNs, and LSTMs are implemented to detect patterns from behavioral data.

## Requirements

To run this code, you'll need the following dependencies:

- Python 3.x
- TensorFlow
- scikit-learn
- pandas
- numpy
- matplotlib
- imbalanced-learn

You can install the required dependencies using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Datasets

The repository contains preprocessed datasets for both **Cyberbullying detection** and **IGD**. The following datasets are included:

### Cyberbullying Datasets:
- `aggression_parsed_dataset.csv`
- `attack_parsed_dataset.csv`
- `kaggle_parsed_dataset.csv`
- `toxicity_parsed_dataset.csv`
- `twitter_parsed_dataset.csv`
- `twitter_racism_parsed_dataset.csv`
- `twitter_sexism_parsed_dataset.csv`
- `youtube_parsed_dataset.csv`

### IGD Dataset:
- `GamingStudy_data.csv` (This is the dataset used for predicting Internet Gaming Disorder based on user behavior data.)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/BryanSJamesDev/IGD-cyberbullying-detection-AI
   cd IGD-cyberbullying-detection-AI
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Code

1. **Cyberbullying Prediction**:
   - Open the `Cyberbullying.ipynb` notebook and run the cells in order to train and evaluate the deep learning models on the provided datasets.

2. **Internet Gaming Disorder Prediction**:
   - Open the `Gamestudy.ipynb` notebook and run the cells to analyze IGD data using models like LSTM and CNN to detect patterns in gaming behavior.

### Datasets Structure

- Place the datasets in the `data/` directory before running the code. The default path for loading datasets is set to this folder.

### Example:

```bash
jupyter notebook Cyberbullying.ipynb
```

## Observed Results

- **Cyberbullying Detection**:
  - CNN and Random Forest models achieve accuracy of around 91% to 93%.
  - The ensemble model yields the best performance with an accuracy of 93%.

- **IGD Detection**:
  - The LSTM model achieves 91.6% accuracy in detecting IGD from gaming behavioral data.

The notebooks will output the model performance metrics, including confusion matrices, precision, recall, F1-scores, and accuracy.
