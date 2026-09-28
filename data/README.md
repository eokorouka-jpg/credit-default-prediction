# Dataset

## Default of Credit Card Clients

This project uses the **Default of Credit Card Clients** dataset from the UCI Machine Learning Repository.

The dataset contains **30,000 customer records** and includes information relating to credit limits, demographic characteristics, repayment status, bill amounts, previous payments, and whether a customer defaulted on payment in the following month.

## Target Variable

The original target variable is:

`default payment next month`

For clarity during the analysis, it was renamed to:

`DEFAULT`

where:

- `0` = No Default
- `1` = Default

The dataset contains:

- **23,364 non-default cases (77.88%)**
- **6,636 default cases (22.12%)**

This class imbalance was considered during model evaluation, with particular attention given to precision, recall, F1-score, and ROC-AUC rather than relying on accuracy alone.

## Dataset Usage

The raw dataset is not included in this repository.

The analysis notebook loads the original Excel (`.xls`) dataset locally. To reproduce the analysis, download the dataset from the UCI Machine Learning Repository and place it in the appropriate local data directory.

## Data Preparation

Key preparation steps included:

- Correcting the Excel header during import
- Renaming the target variable to `DEFAULT`
- Checking for missing values and duplicate records
- Consolidating undocumented/other education categories
- Consolidating the unknown marriage category
- Creating average payment and average bill amount features
- Excluding the customer `ID` from model training

The complete cleaning and preparation process is documented in the project notebook.
