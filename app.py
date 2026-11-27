"""
Classroom Facility Issue Pattern Mining and Maintenance Prioritization
Interactive Decision Support Dashboard (PROTOTYPE) - Agusan del Sur State University

Run:  streamlit run app.py
NOTE: Uses SYNTHETIC sample data (issue_reports_synthetic.csv) for prototyping only.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from analysis import (confusion, load_data, mine_rules, priority_scores, train_classifiers)

st.set_page_config(page_title="ADSSU Facility Issue Dashboard", layout="wide")
st.title("Classroom Facility Issue Dashboard")
st.warning("PROTOTYPE: this dashboard currently uses SYNTHETIC sample data, not real ADSSU records. "
           "Results shown are for demonstration of the system design only.")


@st.cache_data
def get_data(path):
    return load_data(path)


uploaded = st.sidebar.file_uploader("Upload issue reports (CSV)", type="csv")
df_all = get_data(uploaded if uploaded else "issue_reports_synthetic.csv")

# ---------- Filters ----------
st.sidebar.header("Filters")
dmin, dmax = df_all["date_reported"].min().date(), df_all["date_reported"].max().date()
date_range = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
colleges = st.sidebar.multiselect("College", sorted(df_all["college"].unique()))
buildings = st.sidebar.multiselect("Building", sorted(df_all["building"].unique()))
cats = st.sidebar.multiselect("Issue category", sorted(df_all["issue_category"].unique()))
prios = st.sidebar.multiselect("Priority", ["High", "Medium", "Low"])

df = df_all.copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    df = df[(df["date_reported"].dt.date >= date_range[0]) & (df["date_reported"].dt.date <= date_range[1])]
if colleges:
    df = df[df["college"].isin(colleges)]
if buildings:
    df = df[df["building"].isin(buildings)]
if cats:
    df = df[df["issue_category"].isin(cats)]
if prios:
    df = df[df["priority"].isin(prios)]

if df.empty:
    st.error("No records match the selected filters.")
    st.stop()

tabs = st.tabs(["1. Data Exploration", "2. Data Mining Analysis", "3. Interactive Visualization",
                "4. Interpretation & Insights", "5. Decision Support", "6. Reporting"])

# ---------- 1. Data Exploration ----------
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total reports", len(df))
    c2.metric("High priority", int((df["priority"] == "High").sum()))
    c3.metric("Unresolved", int((df["status"] != "Resolved").sum()))
    resolved = df.dropna(subset=["date_resolved"])
    avg_days = (resolved["date_resolved"] - resolved["date_reported"]).dt.days.mean()
    c4.metric("Avg. days to resolve", f"{avg_days:.1f}" if pd.notna(avg_days) else "n/a")
    left, right = st.columns(2)
    left.plotly_chart(px.bar(df["issue_category"].value_counts().reset_index(),
                             x="issue_category", y="count", title="Reports by issue category"),
                      use_container_width=True)
    monthly = df.groupby("month").size().reset_index(name="reports")
    right.plotly_chart(px.line(monthly, x="month", y="reports", title="Reports per month"),
                       use_container_width=True)
    st.subheader("Missing-value summary")
    st.dataframe(df.isna().sum().rename("missing").to_frame())
    with st.expander("Summary statistics and sample rows"):
        st.write(df.describe(include="all", datetime_is_numeric=True) if False else df.describe(include="all"))
        st.dataframe(df.head(20))

# ---------- 2. Data Mining Analysis ----------
with tabs[1]:
    st.subheader("Text classification (80/20 stratified split)")
    metrics, fitted, yte = train_classifiers(df_all)
    st.dataframe(metrics, use_container_width=True)
    best = metrics.sort_values("F1-score", ascending=False).iloc[0]["Model"]
    chosen = st.selectbox("Show confusion matrix for", list(fitted), index=list(fitted).index(best))
    cm = confusion(yte, fitted[chosen][1])
    st.plotly_chart(px.imshow(cm, text_auto=True, title=f"Confusion matrix: {chosen}",
                              labels=dict(x="Predicted", y="Actual")), use_container_width=True)
    st.caption("Classify a new report:")
    new_text = st.text_input("Issue description", "sira ang electric fan sa B-103")
    if new_text:
        st.success(f"Predicted category: **{fitted[chosen][0].predict([new_text])[0]}**")
    st.subheader("Association rules (issues that occur together in the same room and month)")
    c1, c2 = st.columns(2)
    min_sup = c1.slider("Minimum support", 0.01, 0.20, 0.02, 0.01)
    min_conf = c2.slider("Minimum confidence", 0.10, 0.80, 0.20, 0.05)
    rules = mine_rules(df, min_sup, min_conf)
    rules = rules[rules["Lift"] > 1] if not rules.empty else rules
    st.dataframe(rules, use_container_width=True)

# ---------- 3. Interactive Visualization ----------
with tabs[2]:
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(df.groupby(["building", "issue_category"]).size().reset_index(name="reports"),
                           x="building", y="reports", color="issue_category", title="Issues by building"),
                    use_container_width=True)
    c2.plotly_chart(px.bar(df.groupby(["college", "priority"]).size().reset_index(name="reports"),
                           x="college", y="reports", color="priority", title="Priority by college"),
                    use_container_width=True)
    heat = df.pivot_table(index="room", columns="issue_category", values="report_id", aggfunc="count", fill_value=0)
    st.plotly_chart(px.imshow(heat, aspect="auto", title="Room by issue category (drill-down heatmap)"),
                    use_container_width=True)
    room = st.selectbox("Drill down: select a room", sorted(df["room"].unique()))
    st.dataframe(df[df["room"] == room][["report_id", "date_reported", "issue_description",
                                         "issue_category", "priority", "status"]], use_container_width=True)

# ---------- 4. Interpretation & Insights ----------
with tabs[3]:
    top_cat = df["issue_category"].value_counts().idxmax()
    top_room = df["room"].value_counts().idxmax()
    top_bld = df["building"].value_counts().idxmax()
    st.markdown(f"- Most frequently reported category in the selected data: **{top_cat}**.")
    st.markdown(f"- Room with the most reports: **{top_room}**; building with the most reports: **{top_bld}**.")
    r = mine_rules(df, 0.02, 0.2)
    r = r[r["Lift"] > 1] if not r.empty else r
    if not r.empty:
        top = r.iloc[0]
        st.markdown(f"- Strongest association: when **{top['If issue']}** issues are reported in a room, "
                    f"**{top['Then also']}** issues are also reported in about **{top['Confidence']:.0%}** of "
                    f"those room-months (lift {top['Lift']:.2f}).")
    st.info("Interpretations are generated from the data currently loaded and filtered. "
            "With synthetic data they only demonstrate how findings will be presented.")

# ---------- 5. Decision Support ----------
with tabs[4]:
    scores = priority_scores(df)
    st.markdown("**Priority score** = 35% issue volume + 30% high-priority reports + 20% months with issues "
                "+ 15% unresolved issues (each scaled 0 to 1). Weights are adjustable in `analysis.py`.")
    st.dataframe(scores.head(15), use_container_width=True)
    st.plotly_chart(px.bar(scores.head(10), x="room", y="priority_score", color="action",
                           title="Top 10 rooms for maintenance attention"), use_container_width=True)

# ---------- 6. Reporting ----------
with tabs[5]:
    summary = (f"Facility Issue Summary Report\nRecords analyzed: {len(df)}\n"
               f"Date range: {df['date_reported'].min().date()} to {df['date_reported'].max().date()}\n"
               f"Top category: {df['issue_category'].value_counts().idxmax()}\n"
               f"High-priority reports: {int((df['priority'] == 'High').sum())}\n"
               f"Unresolved reports: {int((df['status'] != 'Resolved').sum())}\n")
    st.text(summary)
    st.download_button("Download priority list (CSV)", priority_scores(df).to_csv(index=False),
                       "priority_list.csv", "text/csv")
    st.download_button("Download filtered records (CSV)", df.to_csv(index=False),
                       "filtered_reports.csv", "text/csv")
    st.download_button("Download summary (TXT)", summary, "summary_report.txt")
