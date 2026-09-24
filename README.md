# Customer Segmentation Using Fuzzy C-Means and PCA

This project applies **unsupervised machine learning** to segment telecom customers based on their demographic information, subscribed services, contract details, and billing behavior.

The project implements both **Fuzzy C-Means (FCM)** clustering and **Principal Component Analysis (PCA) from scratch**, rather than relying on ready-made clustering and PCA implementations.

The analysis uses the **IBM Telco Customer Churn dataset**, containing information about more than 7,000 telecom customers.

---

## Project Objective

Customer churn is a major challenge for telecom companies. Instead of directly predicting whether a customer will churn using supervised learning, this project explores the customer population using **unsupervised learning**.

The main objectives are to:

* Discover hidden customer segments.
* Identify groups with similar behavior.
* Model uncertainty in customer membership using Fuzzy C-Means.
* Reduce dimensionality using PCA.
* Compare clustering performance before and after PCA.
* Analyze the relationship between discovered clusters and customer churn.
* Identify customer segments that may require additional retention attention.

> The `Churn` variable is **not used during model training**. It is stored separately and used only after clustering to analyze the churn behavior of the discovered customer segments.

---

## Dataset

The project uses the **Telco Customer Churn** dataset available on Kaggle.

**Original dataset size:**

```text
7,043 customers
21 features
```

The dataset contains customer information in four main categories.

### Demographic Information

* Gender
* Senior citizen status
* Partner
* Dependents

### Services

* Phone service
* Multiple lines
* Internet service
* Online security
* Online backup
* Device protection
* Technical support
* Streaming TV
* Streaming movies

### Contract and Account Information

* Contract type
* Paperless billing
* Payment method

### Billing and Customer History

* Tenure
* Monthly charges
* Total charges

Dataset source:

[Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Project Workflow

The project follows the following pipeline:

```text
Raw Telecom Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Feature Encoding
        │
        ▼
Feature Scaling
        │
        ├──────────────► Fuzzy C-Means
        │                     │
        │                     ▼
        │              Customer Clusters
        │
        ▼
PCA From Scratch
        │
        ▼
Reduced Feature Space
        │
        ▼
Fuzzy C-Means
        │
        ▼
Clustering Evaluation
        │
        ▼
Churn Analysis
```

---

# Data Preprocessing

## Cleaning

`TotalCharges` is converted from an object/string column into a numeric variable.

```python
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
```

Rows containing missing values after conversion are removed.

The customer identifier is also removed because it does not provide useful behavioral information for clustering.

```python
df.dropna(inplace=True)
df.drop(columns=["customerID"], inplace=True)
```

After preprocessing, the analysis contains approximately:

```text
7,032 customers
```

---

## Churn Separation

The churn variable is removed before clustering:

```python
y_churn = df["Churn"]
df = df.drop(columns=["Churn"])
```

This is important because the clustering algorithm must discover customer segments **without knowing the actual churn labels**.

Churn is later used only to interpret the discovered clusters.

---

# Feature Engineering

## Encoding

Binary/categorical variables are converted into numerical representations.

Binary-type variables are encoded using `LabelEncoder`.

```python
le = LabelEncoder()

for col in binary_features:
    df[col] = le.fit_transform(df[col])
```

Variables with multiple categories are converted using one-hot encoding:

```python
df = pd.get_dummies(
    df,
    columns=multi_category_features
)
```

---

## Feature Scaling

The numerical variables are standardized using `StandardScaler`.

```python
numerical_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

scaler = StandardScaler()

df[numerical_features] = scaler.fit_transform(
    df[numerical_features]
)
```

Scaling prevents variables with larger numerical ranges from dominating the distance calculations used by clustering.

---

# PCA From Scratch

One of the main technical components of this project is a custom implementation of **Principal Component Analysis**.

Instead of using:

```python
sklearn.decomposition.PCA
```

PCA is implemented manually.

The implementation includes:

* Mean centering
* Covariance matrix calculation
* Householder transformations
* QR decomposition
* Iterative eigenvalue calculation
* Eigenvector extraction
* Dimensionality reduction
* Data reconstruction
* Explained variance calculation

---

## PCA Implementation

The custom PCA class contains functions for:

```text
cal_norm()
cal_QR()
calculate_eigen()
fit_transform()
inverse()
evr()
```

The covariance matrix is decomposed using repeated **QR iterations** to calculate its eigenvalues and eigenvectors.

The principal components associated with the largest eigenvalues are then selected.

---

# PCA Experiments

Different numbers of principal components were tested.

| Components | Reconstruction MSE | Explained Variance |
| ---------: | -----------------: | -----------------: |
|          2 |           0.139111 |             51.39% |
|          3 |           0.118348 |             58.64% |
|          5 |           0.090948 |             68.22% |

The project selected:

```text
Q = 3 principal components
```

as a compromise between dimensionality reduction, reconstruction error, explained variance, and interpretability.

---

# Fuzzy C-Means From Scratch

The second major component of the project is a custom implementation of **Fuzzy C-Means clustering**.

Unlike traditional clustering methods such as K-Means, Fuzzy C-Means does not force every observation to belong completely to one cluster.

Instead, every customer receives a **membership probability/degree for each cluster**.

For example, a customer could have membership values similar to:

```text
Cluster 0: 0.10
Cluster 1: 0.25
Cluster 2: 0.65
```

This customer would primarily belong to Cluster 2, while still sharing characteristics with the other customer segments.

This makes Fuzzy C-Means useful when customer behavior does not fall into perfectly separated categories.

---

## Fuzzy C-Means Algorithm

The custom implementation performs the following steps:

1. Initialize a random membership matrix.
2. Calculate fuzzy cluster centroids.
3. Calculate customer-to-centroid distances.
4. Update membership values.
5. Compare the new and previous membership matrices.
6. Continue until convergence or the maximum number of iterations is reached.

The primary configuration used is:

```text
Fuzziness parameter (m): 2
Maximum iterations:      150
Convergence tolerance:   1e-5
```

---

# Selecting the Number of Clusters

Different numbers of clusters were evaluated:

```text
K = 2
K = 3
K = 4
```

Two fuzzy clustering metrics were used.

### Fuzzy Partition Coefficient — FPC

FPC measures the strength or crispness of cluster membership.

In general:

```text
Higher FPC → clearer membership structure
```

### Partition Entropy — PE

Partition Entropy measures uncertainty in cluster membership.

In general:

```text
Lower PE → less membership uncertainty
```

The recorded experiment produced:

|  K |    FPC | Partition Entropy |
| -: | -----: | ----------------: |
|  2 | 0.5146 |            0.6784 |
|  3 | 0.4120 |            0.9853 |
|  4 | 0.3223 |            1.2529 |

Although `K = 2` achieved the strongest numerical FPC/PE values in this run, the project selected:

```text
K = 3
```

to obtain a more expressive customer segmentation and maintain a useful balance between cluster clarity and business interpretability.

---

# Effect of PCA on Fuzzy C-Means

An important experiment in the project compares Fuzzy C-Means clustering **before and after PCA**.

## Before PCA

```text
Iterations: 144
FPC:        0.4120
PE:         0.9853
```

## After PCA

Using three principal components:

```text
Iterations: 27
FPC:        0.7021
PE:         0.5497
```

### Comparison

| Metric            | Before PCA | After PCA |
| ----------------- | ---------: | --------: |
| Iterations        |        144 |        27 |
| FPC               |     0.4120 |    0.7021 |
| Partition Entropy |     0.9853 |    0.5497 |

These experimental results show a substantial improvement after dimensionality reduction.

### Observed Improvements

**Convergence**

```text
144 → 27 iterations
```

The reduced PCA representation allowed the algorithm to converge considerably faster.

**Fuzzy Partition Coefficient**

```text
0.4120 → 0.7021
```

The higher FPC indicates stronger cluster membership assignments.

**Partition Entropy**

```text
0.9853 → 0.5497
```

The lower entropy indicates less ambiguity between cluster memberships.

Overall, the experiment suggests that PCA produced a lower-dimensional representation that was more suitable for the Fuzzy C-Means clustering procedure used in this notebook.

---

# Customer Churn Analysis

After clustering, the original churn labels are added back to the dataset for analysis.

The churn variable was **not used to create the clusters**.

The observed churn proportions in the three-cluster solution were:

|   Cluster | Stayed |    Churned |
| --------: | -----: | ---------: |
| Cluster 0 | 83.83% | **16.17%** |
| Cluster 1 | 92.57% |  **7.43%** |
| Cluster 2 | 55.78% | **44.22%** |

The analysis indicates substantial differences in churn behavior between the discovered customer groups.

### Cluster 0

```text
Churn rate ≈ 16.2%
```

This segment shows relatively low-to-moderate churn.

### Cluster 1

```text
Churn rate ≈ 7.4%
```

This segment has the lowest observed churn rate.

### Cluster 2

```text
Churn rate ≈ 44.2%
```

This segment contains the highest proportion of customers who churned and may therefore warrant further investigation for customer-retention analysis.

---

# Visualization

The notebook contains several visualizations used to evaluate the clustering process.

## Membership Distributions

Membership histograms are generated for different values of `K` to examine how strongly customers belong to each fuzzy cluster.

## PCA Cluster Visualization

The first two principal components are visualized using a scatter plot:

```python
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1]
)
```

This provides a two-dimensional representation of the customer segmentation.

## Churn by Cluster

A stacked bar chart is used to compare churn proportions across the discovered clusters.

These visualizations help connect the mathematical clustering results with interpretable customer behavior.

---

# Technologies Used

The project uses:

```text
Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
KaggleHub
Jupyter Notebook / Google Colab
```

The main machine-learning techniques include:

```text
Unsupervised Learning
Fuzzy C-Means Clustering
Principal Component Analysis
Dimensionality Reduction
Customer Segmentation
Cluster Evaluation
Feature Scaling
Categorical Encoding
```

---

# Key Technical Features

A major focus of this project is implementing important machine-learning algorithms manually.

### PCA

Implemented from scratch using:

```text
Covariance Matrix
Householder Transformation
QR Decomposition
Eigenvalues
Eigenvectors
Principal Components
Reconstruction Error
Explained Variance
```

### Fuzzy C-Means

Implemented from scratch using:

```text
Random Membership Initialization
Fuzzy Membership Matrix
Weighted Centroids
Euclidean Distance
Iterative Membership Updates
Convergence Checking
Fuzzy Partition Coefficient
Partition Entropy
```

This provides a deeper understanding of how the algorithms work internally rather than treating them as black-box library functions.

---

# How to Run the Project

Clone the repository:

```bash
git clone <your-repository-url>
cd <repository-name>
```

Install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn kagglehub jupyter
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
Unsupervised_learning_project_v2.ipynb
```

The notebook retrieves the Telco Customer Churn dataset using `kagglehub`.

---

# Project Structure

```text
customer-segmentation/
│
├── Unsupervised_learning_project_v2.ipynb
│
└── README.md
```

---

# Key Results

The main findings of the project are:

* Successfully implemented **PCA from scratch**.
* Implemented eigenvalue decomposition using **Householder transformations and QR iterations**.
* Successfully implemented **Fuzzy C-Means from scratch**.
* Identified three customer segments.
* Reduced the customer representation to three principal components.
* Achieved approximately **58.64% explained variance** using three components.
* Improved FPC from **0.4120 to 0.7021** after PCA.
* Reduced Partition Entropy from **0.9853 to 0.5497** after PCA.
* Reduced FCM convergence from **144 iterations to 27 iterations** after PCA.
* Identified a cluster with an observed churn proportion of approximately **44.22%**.

---

# Limitations

Several limitations should be considered when interpreting the results.

The Fuzzy C-Means algorithm uses random membership initialization, so exact clustering results may vary between runs unless a random seed is specified.

The selection of three clusters includes an interpretability consideration; the FPC and Partition Entropy results alone were strongest for `K = 2` in the recorded experiment.

The customer profiles assigned to individual clusters should be validated through additional feature-level cluster profiling before making strong business conclusions about the characteristics of each segment.

---

# Future Improvements

Possible extensions include:

* Set a fixed random seed for fully reproducible FCM results.
* Save the final membership matrix and cluster assignments.
* Create detailed profiles of each cluster using original customer features.
* Compare Fuzzy C-Means with K-Means.
* Compare the custom PCA implementation with `sklearn.decomposition.PCA`.
* Add Silhouette Score for hard-label comparison.
* Apply t-SNE or UMAP for additional visualization.
* Implement additional clustering techniques such as DBSCAN.
* Analyze centroid characteristics using original feature units.
* Build an interactive customer-segmentation dashboard.
* Develop retention recommendations based on verified cluster characteristics.

---

# Skills Demonstrated

This project demonstrates experience with:

**Python • Unsupervised Learning • Machine Learning • Customer Segmentation • Fuzzy C-Means • PCA • Dimensionality Reduction • Linear Algebra • QR Decomposition • Feature Engineering • Data Preprocessing • Data Visualization • Cluster Evaluation**

---

# References

**Dataset**

IBM Telco Customer Churn Dataset
Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

**Scikit-learn — StandardScaler**

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

**Pandas**

https://pandas.pydata.org/docs/

**NumPy**

https://numpy.org/doc/
