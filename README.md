# NHS A&E Breach Predictor

A machine learning system that predicts which NHS hospitals are at risk of experiencing high-pressure month, defined as more than 30% of A&E patients waiting over 4 hours. Built on 36 months of real NHS England data covering 122 hospitals per month.

---

## Problem Statement

NHS hospitals are required to treat, admit, or discharge 95% of A&E patients within 4 hours. In practice, nearly every hospital misses this target. The real operational question is not whether a hospital will breach, but how severely. This model predicts which hospitals will experience high-pressure months (breach rate above 30%), enabling early intervention before the situation becomes critical.

---

## Results

| Model | F1 Score | Recall (Breached) | AUC |
|---|---|---|---|
| Logistic Regression (baseline) | 0.748 | 0.89 | — |
| Random Forest (final) | 0.939 | 0.97 | 0.888 |

The final model correctly identifies 97% of high-pressure hospital months. Only 19 out of 649 real crises were missed across the test period.

---

## Dataset

Source: NHS England A&E Attendances and Emergency Admissions statistics
Link: https://www.england.nhs.uk/statistics/statistical-work-areas/ae-waiting-times-and-activity/
Licence: Open Government Licence v3.0
Coverage: April 2022 to March 2025 (36 monthly CSV files)
Scope: 122 Type 1 (major) A&E hospitals per month after filtering

---

## Project Structure

```
nhs-ae-analyser/
│
├── data/
│   ├── raw/                    # Original NHS CSVs (not tracked in git)
│   └── processed/              # Cleaned and engineered dataset
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis (single month)
│   ├── 02_features.ipynb       # Full dataset loading and feature engineering
│   └── 03_modelling.ipynb      # Model experimentation and evaluation
│
├── src/
│   ├── train.py                # Production training script
│   └── evaluate.py             # Model evaluation with TimeSeriesSplit
│
├── models/
│   └── random_forest.pkl       # Saved trained model
│
├── outputs/
│   └── figures/                # All EDA and evaluation charts
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Feature Engineering

Three features were engineered from raw NHS columns:

**breach_rate** : proportion of Type 1 patients waiting over 4 hours (used for target variable creation only, excluded from model features to prevent leakage)

**admission_rate** : proportion of Type 1 patients requiring hospital admission. Captures how many patients needed a bed beyond emergency treatment.

**severe_wait_ratio** : proportion of patients waiting 12 or more hours after a decision to admit. A direct measure of systemic hospital pressure.

Raw attendance volume (type1_att) was retained as a feature despite weak direct correlation with breach rate. Random Forest uses it in combination with other features rather than in isolation.

---

## Key Findings from EDA

The typical NHS hospital in this period has a 42% breach rate — meaning 42% of patients wait over 4 hours. This is systemic, not isolated to a few struggling hospitals.

Attendance volume alone does not predict breach rate. A hospital processing 20,000 patients per month can perform better than one processing 8,000. The combination of admission pressure and severe wait patterns is far more predictive.

One TOTAL aggregate row was embedded in the raw data alongside real hospital records. Specialist hospitals with zero Type 1 attendances were also present. Both required filtering before analysis.

---

## Modelling Decisions

**Why Random Forest over Logistic Regression**
The relationship between hospital pressure indicators and breach risk is non-linear. Logistic regression draws a linear boundary and misses complex interactions. Random Forest with 100 trees captures these interactions and improves F1 from 0.748 to 0.939.

**Why TimeSeriesSplit over random cross-validation**
NHS data is temporal. January performance is correlated with February performance. Random splitting causes data leakage — the model trains on data from the same time period it is tested on. TimeSeriesSplit ensures the model always trains on past data and is tested on future data.

**Why recall over precision as primary metric**
Missing a high-pressure hospital (false negative) has patient safety consequences. Wrongly flagging a healthy hospital (false positive) wastes management attention. Given this asymmetry, recall on the breached class was prioritised.

**Target leakage discovered and fixed**
Initial F1 score of 0.989 was caused by breach_rate being included as a feature. The target variable breached was created directly from breach_rate, giving the model the answer during training. Removing breach_rate dropped the score to a realistic 0.939.

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/isha-atif-dev/nhs-ae-analyser
cd nhs-ae-analyser
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download NHS data**
Download monthly A&E CSV files from the NHS England link above. Place all files in data/raw/.

**4. Run feature engineering**
Open and run notebooks/02_features.ipynb. This generates data/processed/nhs_processed.csv.

**5. Train the model**
```bash
python src/train.py
```

**6. Evaluate the model**
```bash
python src/evaluate.py
```

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
imbalanced-learn
python-dotenv
```

---

## Author

Isha Atif
MRes Applied Artificial Intelligence, University of Greater Manchester
Data Operations Analyst, BeautyZone London
GitHub: github.com/isha-atif-dev
LinkedIn: linkedin.com/in/isha-atif

---

## Licence

Data sourced from NHS England under the Open Government Licence v3.0.
Code in this repository is available for personal and educational use.
