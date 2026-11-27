# Classroom Facility Issue Pattern Mining Dashboard (Prototype)

Interactive decision support dashboard for the Capstone Project 1 proposal:
**Classroom Facility Issue Pattern Mining and Maintenance Prioritization Using Text Classification
and Association Rule Mining with an Interactive Decision Support Dashboard for Agusan del Sur State University.**

> **PROTOTYPE NOTICE:** the dataset in this repository (`issue_reports_synthetic.csv`) is
> **synthetic**. It is not real ADSSU data. It only demonstrates the system design. Real, anonymized
> records from General Services will be used in Capstone 2.

## What it does
| Module | Content |
| --- | --- |
| 1. Data Exploration | Dataset overview, key figures, distributions, trends, missing-value summary |
| 2. Data Mining Analysis | Naive Bayes, Logistic Regression and Linear SVM comparison (accuracy, precision, recall, F1), confusion matrix, top terms per category, new-report classifier, Apriori association rules (support, confidence, lift) |
| 3. Interactive Visualization | Sidebar filters, comparison charts, heatmap, drill-down from building to room to reports |
| 4. Interpretation & Insights | Plain-language findings, rising issue types, meaning of association rules |
| 5. Decision Support | Transparent room priority score, recommended action, suggested management action |
| 6. Reporting | Downloadable summary (TXT) and CSV files (priority list, filtered records, rules) |

## Files
- `app.py` - Streamlit dashboard
- `analysis.py` - text classification, association rules, priority score
- `generate_data.py` - creates the synthetic dataset (`python generate_data.py`)
- `issue_reports_synthetic.csv` - synthetic sample data (about 430 records, Aug 2023 to Jul 2026)
- `requirements.txt` - Python dependencies

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Data columns
`report_id, date_reported, college, building, room, issue_description, issue_category, priority, status, date_resolved`

Issue categories: Electrical, Furniture, Ventilation, Audio-Visual, Plumbing, Structural, Other.
Priority levels: High, Medium, Low.

## Priority score (rule-based, transparent)
30% issue frequency + 25% high-priority reports + 20% recurrence (months with issues)
+ 10% unresolved issues + 15% association-rule strength. Weights can be changed in `analysis.py`.
