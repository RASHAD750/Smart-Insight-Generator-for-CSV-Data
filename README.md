#  API-Free CSV Insight Generator

A fully **local, rule-based data analysis and insight generation system** built using **Streamlit**, **Scikit-learn**, and **Pandas**.

This project performs:

-  Automated Exploratory Data Analysis (EDA)
-  Correlation Detection
-  Feature Importance (Random Forest)
-  Rule-Based Insight Summary Generation
-  Data Quality Assessment
-  Interactive Q&A
-  Downloadable PDF Insight Report

 No external APIs  
 No OpenAI / LLM dependency  
 Fully offline and reproducible  

---

#  Project Overview

This application allows users to upload a CSV file and automatically:

1. Analyze dataset structure
2. Detect missing values
3. Generate descriptive statistics
4. Identify top correlations
5. Compute feature importance
6. Generate executive-level insight reports
7. Answer structured business questions
8. Export findings to PDF

It is ideal for:

- Data Science students
- Business Analysts
- Academic projects
- Portfolio projects
- Offline environments

---

#  Project Architecture

```
.
├── app.py               # Streamlit UI and orchestration
├── data_analyzer.py     # EDA + Feature Importance Engine
├── rule_generator.py    # Rule-Based Insight Engine
├── requirements.txt     # Dependencies
└── README.md
```

---

#  Core Modules Explained

---

## 1️. data_analyzer.py

Handles statistical computation.

###  perform_eda(df)

Performs:

- Dataset shape detection
- Missing value analysis
- Descriptive statistics
- Top 5 absolute correlations

Returns structured dictionary:

```python
{
    "shape": "...",
    "null_report": {...},
    "descriptive_stats": {...},
    "top_correlations": [...]
}
```

---

###  get_feature_importance(df, target_col)

Uses:

- RandomForestRegressor
- Returns Top 5 most important features

```python
[
    {"feature": "X1", "importance": 0.45},
    ...
]
```

✔ Works only on numeric columns  
✔ Drops NA values  
✔ Includes safety validation  

---

## 2️. rule_generator.py

Contains rule-based business logic engine.

###  generate_rule_based_summary()

Creates structured insight report:

```python
{
    "executive_summary": "",
    "key_relationships": [],
    "data_quality_assessment": "",
    "strategic_recommendation": ""
}
```

### Rules Applied:

- Correlation ≥ 0.5 → moderate/strong insight
- Missing > 5% → major data issue
- Top feature importance → strategic focus

---

###  answer_question()

Handles rule-based Q&A.

Supported topics:
- Feature impact
- Correlations
- Data quality

Example questions:
- “Which feature affects the target most?”
- “What is the strongest correlation?”
- “Are there missing values?”

---

## 3️. app.py (Streamlit Interface)

Provides interactive UI with 3 tabs:

---

###  Tab 1 – Auto EDA & Charts

- Data preview
- Null report
- Correlation heatmap
- Feature importance bar chart

---

###  Tab 2 – Insight Summary

- Executive summary
- Key relationships
- Data quality report
- Strategic recommendation
- PDF download button

---

###  Tab 3 – Ask Your Data

- Rule-based chatbot
- Session state management
- Structured business Q&A

---

#  PDF Report Generation

Uses:

- fpdf2
- Custom PDF class
- Text sanitization for encoding safety

PDF Sections:

1. Executive Summary
2. Key Relationships
3. Data Quality Assessment
4. Strategic Recommendation
5. Data Snapshot

---

#  Requirements

```
streamlit
pandas
plotly
numpy
fpdf2
scikit-learn
```

---

#  Installation Guide

## Step 1 — Clone Repository

```bash
git clone https://github.com/yourusername/api-free-insight-generator.git
cd api-free-insight-generator
```

## Step 2 — Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate:

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

#  Run the Application

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

#  How It Works (Execution Flow)

1. User uploads CSV
2. perform_eda() runs
3. Correlations computed
4. User selects target column
5. Random Forest computes feature importance
6. generate_rule_based_summary() builds insights
7. PDF report created
8. Q&A engine responds using precomputed stats

---

#  Rule-Based vs AI-Based Approach

| Feature | This Project | AI/LLM Approach |
|----------|-------------|----------------|
| Internet Required | ❌ | ✅ |
| Cost | Free | Paid |
| Transparency | High | Low |
| Reproducibility | High | Variable |
| Deterministic | Yes | No |

---

#  Limitations

- Works only with CSV files
- Feature importance requires numeric target
- No NLP understanding beyond simple keywords
- No categorical encoding
- Linear correlation only
- Random Forest assumes regression target

---

#  Future Improvements

- Add classification support
- Add SHAP explainability
- Add categorical encoding
- Add outlier detection
- Add automated preprocessing
- Add dashboard export
- Deploy on Streamlit Cloud

---

#  Example Use Case

Upload a Sales dataset:

| Sales | Marketing | Price | Discount |
|-------|----------|-------|----------|

The system will:

- Detect strongest correlation
- Identify most influential driver of Sales
- Flag missing values
- Recommend strategic focus
- Generate downloadable executive PDF

---
