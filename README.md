# ADSSU Classroom Facility Issue Dashboard (Prototype)

Capstone 1 prototype for: Classroom Facility Issue Pattern Mining and Maintenance
Prioritization Using Text Classification and Association Rule Mining with an Interactive
Decision Support Dashboard for Agusan del Sur State University.

## IMPORTANT
`issue_reports_synthetic.csv` is SYNTHETIC data created by `generate_data.py` for design
and demonstration only. It is not real ADSSU data. Do not present its results as findings.
Replace it with anonymized records approved by General Services (same columns) for Capstone 2.

## Run
    pip install -r requirements.txt
    python generate_data.py      # optional: regenerate sample data
    streamlit run app.py

## Files
- generate_data.py: creates the synthetic sample dataset
- analysis.py: text classification, association rules, priority score
- app.py: Streamlit dashboard (six required modules)
- issue_reports_synthetic.csv: sample data (914 rows)

## Columns
report_id, date_reported, college, building, room, issue_description, issue_category,
priority, status, date_resolved
