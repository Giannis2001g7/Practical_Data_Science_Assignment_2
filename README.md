Master in Data Science AUEB 2024-2025

Practical Data Science Assignment 2

Ioannis Papadopoulos f3352409

FOOD HAZARD DETECTION CHALLENGE

This repository contains a machine learning pipeline for detecting food hazards and classifying associated products. The project leverages both basic (Logistic Regression) and advanced (XGBoost) machine learning models, optimizing for textual data analysis using TF-IDF vectorization. It includes end-to-end solutions for both tasks (ST1 and ST2) of the Food Hazard Detection Challenge. The solution is designed to run seamlessly on platforms like Google Colab.

Requirements:
The pipeline requires the following Python libraries:

pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
imblearn
spacy
nltk
zipfile
Install the required libraries in your Colab environment using pip install. Additionally, ensure that the Spacy language model en_core_web_sm is downloaded:
!python -m spacy download en_core_web_sm

Datasets
The project expects the following input files:

incidents_train.csv: Contains training data with features title, text, and labels hazard-category and product-category.
incidents.csv: Testing dataset with title and text features for prediction.
Both files should be uploaded into the Colab environment when prompted.

Key Features:
Data Cleaning:
Preprocessing includes:
Lowercasing text.
Removing stopwords and punctuation.
Lemmatizing using Spacy NLP.
Applied to title and text columns for both training and test datasets.

Exploratory Data Analysis (EDA):
Visualized class distributions for hazard-category and product-category using bar plots.
Generated Word Clouds for title and text to understand common terms.

ST1: Hazard and Product Category Prediction
Predicts hazard-category and product-category independently.
Models used:
Logistic Regression
XGBoost
Applied TF-IDF vectorization to title and text features.
Applied K-Fold Cross Validation.
Imbalance handled with SMOTE.
Evaluated models using macro-F1 score and selected the best-performing model for each category and feature.
Outputs:
Predictions for each feature-model combination saved in submission_ST1.zip.
Best-performing combination: XGBoost for hazard-category and Logistic Regression for product-category using the title feature.

ST2: Hazard and Product Classification
Combines the prediction of hazards and products into a unified task.
Preprocessing:
Rare class handling by merging categories with fewer than a threshold number of samples into an other category.
Models used:
Logistic Regression
XGBoost
Utilized TF-IDF vectorization with dynamic adjustments to SMOTE parameters for handling class imbalance.
Outputs:
Predictions for each feature-model combination saved in predictions_ST2.zip.
Best-performing system: Logistic Regression using the title feature.

Evaluation Metrics
For both tasks, the models were evaluated using:

Macro-F1 Score: The primary evaluation metric.

Results
Task	Model	Feature	Metric	Score	Ranking
ST1	XGBoost (Hazard), Logistic Regression (Product)	Title	Macro-F1	score:0.63	leaderbord:31st
ST2	Logistic Regression (Hazard & Product)	Title	Macro-F1	score:0.34	leaderbord:17th

How to Use
Upload Required Datasets
Upload incidents_train.csv and incidents.csv when prompted in the Colab environment.

Run the Code
Execute all code cells sequentially to:

Clean the data.
Train and evaluate models.
Generate predictions and output files.
Outputs

ST1 Results: Saved in submission_ST1.zip.
ST2 Results: Saved in predictions_ST2.zip.
Final merged submission: submission.csv.

Key Functions
Data Preprocessing
clean_text(text): Cleans and lemmatizes text using Spacy NLP.
handle_rare_classes(data, target, threshold): Groups rare classes into an other category.
Model Training and Evaluation
train_model_st1: Trains models for ST1 tasks with TF-IDF vectorization and SMOTE.
train_model_st2: Trains models for ST2 tasks with dynamic SMOTE adjustments.
compute_score: Custom scoring for combined hazard and product predictions (ST1).
Prediction
predict_test_data: Predicts labels for ST1 task.
predict_test_data_st2: Predicts labels for ST2 task.

Output Structure
submission_ST1/: Predictions for ST1 task.
predictions_ST2/: Predictions for ST2 task.
submission.csv: Final merged predictions.

References:
 • ChatGPT: For quick assistance, code help, debugging, and guidance on
 various data science topics.
