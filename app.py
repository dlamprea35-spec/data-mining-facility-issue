"""
Classroom Facility Issue Pattern Mining and Maintenance Prioritization
Interactive Decision Support Dashboard (PROTOTYPE) - Agusan del Sur State University

Run:  streamlit run app.py
NOTE: Uses SYNTHETIC sample data (issue_reports_synthetic.csv) for prototyping only.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from analysis import (category_trend, confusion, load_data, mine_rules,
                      priority_scores, top_terms, train_classifiers)

st.set_page_config(page_title="ADSSU Facility Issue Dashboard", layout="wide")
st.title("Classroom Facility Issue Dashboard")
st.caption("Agusan del Sur State University | Pattern Mining and Maintenance Prioritization")
st.warning("PROTOTYPE: this dashboard currently uses SYNTHETIC sample data, not real ADSSU records. "
           "Results shown are for demonstration of the system design only.")

REQUIRED = ["report_id", "date_reported", "college", "building", "room",
            "issue_description", "issue_category", "priority", "status", "date_resolved"]


# ---------- small helpers (work on both old and new Streamlit versions) ----------
def show_df(data, where=st, **kw):
    try:
        where.dataframe(data, width="stretch", **kw)
    except Exception:
        where.dataframe(data, use_container_width=True, **kw)


def show_plot(fig, where=st):
    try:
        where.plotly_chart(fig, width="stretch")
    except Exception:
        where.plotly_chart(fig, use_container_width=True)


@st.cache_data
def get_data(path):
    return load_data(path)


@st.cache_resource
def get_models(data):
    return train_classifiers(data)


# ---------- Data loading ----------
uploaded = st.sidebar.file_uploader("Upload issue reports (CSV)", type="csv")
try:
    df_all = get_data(uploaded if uploaded else "issue_reports_synthetic.csv")
except Exception as e:
    st.error(f"Could not read the CSV file: {e}")
    st.stop()
missing_cols = [c for c in REQUIRED if c not in df_all.columns]
if missing_cols:
    st.error(f"The CSV is missing required columns: {', '.join(missing_cols)}")
    st.stop()

# ---------- Filters (leave empty = include all) ----------
st.sidebar.header("Filters")
dmin, dmax = df_all["date_reported"].min().date(), df_all["date_reported"].max().date()
date_range = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
colleges = st.sidebar.multiselect("College", sorted(df_all["college"].unique()))
buildings = st.sidebar.multiselect("Building", sorted(df_all["building"].unique()))
room_options = (sorted(df_all[df_all["building"].isin(buildings)]["room"].unique())
                if buildings else sorted(df_all["room"].unique()))
rooms = st.sidebar.multiselect("Room", room_options)
cats = st.sidebar.multiselect("Issue category", sorted(df_all["issue_category"].unique()))
prios = st.sidebar.multiselect("Priority", ["High", "Medium", "Low"])
st.sidebar.caption("Leave a filter empty to include everything.")

df = df_all.copy()
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    df = df[(df["date_reported"].dt.date >= date_range[0]) & (df["date_reported"].dt.date <= date_range[1])]
if colleges:
    df = df[df["college"].isin(colleges)]
if buildings:
    df = df[df["building"].isin(buildings)]
if rooms:
    df = df[df["room"].isin(rooms)]
if cats:
    df = df[df["issue_category"].isin(cats)]
if prios:
    df = df[df["priority"].isin(prios)]

if df.empty:
    st.error("No records match the selected filters.")
    st.stop()

# association rules for the currently filtered data (default thresholds)
rules_default = mine_rules(df, 0.02, 0.2)
rules_default = rules_default[rules_default["Lift"] > 1] if not rules_default.empty else rules_default

tabs = st.tabs(["1. Data Exploration", "2. Data Mining Analysis", "3. Interactive Visualization",
                "4. Interpretation & Insights", "5. Decision Support", "6. Reporting"])

# ---------- 1. Data Exploration ----------
with tabs[0]:
    st.subheader("Dataset overview")
    o1, o2, o3 = st.columns(3)
    o1.metric("Records (all data)", len(df_all))
    o2.metric("Columns", df_all.shape[1])
    o3.metric("Coverage", f"{dmin} to {dmax}")

    st.subheader("Key figures (filtered data)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total reports", len(df))
    c2.metric("High priority", int((df["priority"] == "High").sum()))
    c3.metric("Unresolved", int((df["status"] != "Resolved").sum()))
    resolved = df.dropna(subset=["date_resolved"])
    avg_days = (resolved["date_resolved"] - resolved["date_reported"]).dt.days.mean()
    c4.metric("Avg. days to resolve", f"{avg_days:.1f}" if pd.notna(avg_days) else "n/a")

    left, right = st.columns(2)
    by_cat = df["issue_category"].value_counts().rename_axis("issue_category").reset_index(name="reports")
    show_plot(px.bar(by_cat, x="issue_category", y="reports", title="Reports by issue category"), left)
    monthly = df.groupby("month").size().reset_index(name="reports")
    show_plot(px.line(monthly, x="month", y="reports", title="Reports per month"), right)

    l2, r2 = st.columns(2)
    by_bld = df["building"].value_counts().rename_axis("building").reset_index(name="reports")
    show_plot(px.bar(by_bld, x="building", y="reports", title="Reports by building"), l2)
    by_pri = df["priority"].value_counts().rename_axis("priority").reset_index(name="reports")
    show_plot(px.pie(by_pri, names="priority", values="reports", title="Priority distribution"), r2)

    st.subheader("Missing-value summary")
    show_df(df.isna().sum().rename("missing values").to_frame())
    st.caption("Blank date_resolved is expected for reports that are not yet resolved.")
    with st.expander("Summary statistics and sample rows"):
        show_df(df.describe(include="all").astype(str))
        show_df(df.head(20))

# ---------- 2. Data Mining Analysis ----------
with tabs[1]:
    st.subheader("Text classification (80/20 stratified split)")
    st.caption("Models are trained on all loaded records so results stay stable while filters change.")
    metrics, fitted, yte = get_models(df_all)
    show_df(metrics)
    best = metrics.sort_values("F1-score", ascending=False).iloc[0]["Model"]
    st.markdown(f"Best model by weighted F1-score: **{best}**.")
    chosen = st.selectbox("Show confusion matrix for", list(fitted), index=list(fitted).index(best))
    cm = confusion(yte, fitted[chosen][1])
    show_plot(px.imshow(cm, text_auto=True, title=f"Confusion matrix: {chosen}",
                        labels=dict(x="Predicted", y="Actual")))

    st.markdown("**Most influential terms per category (Logistic Regression)**")
    show_df(top_terms(fitted["Logistic Regression"][0]), hide_index=True)

    st.markdown("**Classify a new report**")
    new_text = st.text_input("Issue description", "sira ang electric fan sa B-103")
    if new_text:
        st.success(f"Predicted category ({chosen}): **{fitted[chosen][0].predict([new_text])[0]}**")

    st.subheader("Association rules (issues that occur together in the same room and month)")
    c1, c2 = st.columns(2)
    min_sup = c1.slider("Minimum support", 0.01, 0.20, 0.02, 0.01)
    min_conf = c2.slider("Minimum confidence", 0.10, 0.80, 0.20, 0.05)
    rules = mine_rules(df, min_sup, min_conf)
    rules = rules[rules["Lift"] > 1] if not rules.empty else rules
    if rules.empty:
        st.info("No rules meet the selected thresholds. Lower the minimum support or confidence.")
    else:
        show_df(rules, hide_index=True)
        st.caption("Support = share of room-months with both issues. Confidence = chance the second issue "
                   "appears when the first does. Lift above 1 = the issues co-occur more than by chance.")

# ---------- 3. Interactive Visualization ----------
with tabs[2]:
    st.caption("All charts respond to the filters in the sidebar.")
    c1, c2 = st.columns(2)
    show_plot(px.bar(df.groupby(["building", "issue_category"]).size().reset_index(name="reports"),
                     x="building", y="reports", color="issue_category", title="Issues by building"), c1)
    show_plot(px.bar(df.groupby(["college", "priority"]).size().reset_index(name="reports"),
                     x="college", y="reports", color="priority", title="Priority by college"), c2)
    trend = df.groupby(["month", "issue_category"]).size().reset_index(name="reports")
    show_plot(px.line(trend, x="month", y="reports", color="issue_category",
                      title="Monthly trend by issue category"))
    heat = df.pivot_table(index="room", columns="issue_category", values="report_id",
                          aggfunc="count", fill_value=0)
    show_plot(px.imshow(heat, aspect="auto", title="Room by issue category (heatmap)"))

    st.subheader("Drill-down: building > room > individual reports")
    d1, d2 = st.columns(2)
    bld = d1.selectbox("Building", sorted(df["building"].unique()))
    room = d2.selectbox("Room", sorted(df[df["building"] == bld]["room"].unique()))
    sel = df[(df["building"] == bld) & (df["room"] == room)]
    st.write(f"{len(sel)} report(s) in {room}")
    show_df(sel[["report_id", "date_reported", "issue_description", "issue_category",
                 "priority", "status"]], hide_index=True)

# ---------- 4. Interpretation & Insights ----------
with tabs[3]:
    top_cat = df["issue_category"].value_counts().idxmax()
    top_room = df["room"].value_counts().idxmax()
    top_bld = df["building"].value_counts().idxmax()
    st.markdown("**What the data shows**")
    st.markdown(f"- Most frequently reported category: **{top_cat}**.")
    st.markdown(f"- Room with the most reports: **{top_room}**; building with the most reports: **{top_bld}**.")
    tr = category_trend(df)
    rising = tr[tr["Change"] > 0]
    if not rising.empty:
        names = ", ".join(rising["Category"].head(3))
        st.markdown(f"- Issue types rising in the latest period compared with the previous one: **{names}**.")
    else:
        st.markdown("- No issue category increased in the latest period compared with the previous one.")
    show_df(tr, hide_index=True)

    st.markdown("**What the patterns mean**")
    if rules_default.empty:
        st.markdown("- No strong co-occurring issues were found in the filtered data.")
    else:
        seen, shown = set(), 0
        for _, r in rules_default.iterrows():
            key = frozenset([r["If issue"], r["Then also"]])
            if key in seen:
                continue
            seen.add(key)
            st.markdown(f"- When **{r['If issue']}** is reported in a room, **{r['Then also']}** is also "
                        f"reported in about **{r['Confidence']:.0%}** of those room-months (lift {r['Lift']:.2f}). "
                        f"General Services can inspect both systems in one visit.")
            shown += 1
            if shown == 3:
                break
    st.info("Interpretations are generated from the data currently loaded and filtered. "
            "With synthetic data they only demonstrate how findings will be presented.")

# ---------- 5. Decision Support ----------
with tabs[4]:
    scores = priority_scores(df, rules_default)
    st.markdown("**Priority score** = 30% issue frequency + 25% high-priority (urgent) reports + "
                "20% recurrence (months with issues) + 10% unresolved issues + 15% association-rule strength "
                "(each scaled 0 to 1). Weights are adjustable in `analysis.py`.")
    cols = ["rank", "building", "room", "priority_score", "action", "top_issue", "total_issues",
            "high_priority", "months_with_issues", "open_issues", "rule_strength",
            "co_occurring_issues", "suggested_action"]
    show_df(scores[cols].head(15), hide_index=True)
    show_plot(px.bar(scores.head(10), x="room", y="priority_score", color="action",
                     title="Top 10 rooms for maintenance attention"))
    st.markdown("**Why is a room ranked high?**")
    pick = st.selectbox("Select a room", scores["room"].tolist())
    row = scores[scores["room"] == pick].iloc[0]
    st.markdown(f"- **{pick}** ranks **#{row['rank']}** with a score of **{row['priority_score']}** "
                f"({row['action']}).")
    st.markdown(f"- {int(row['total_issues'])} reports, {int(row['high_priority'])} high priority, "
                f"issues in {int(row['months_with_issues'])} different months, {int(row['open_issues'])} still open.")
    if row["co_occurring_issues"]:
        st.markdown(f"- Co-occurring issue types in this room: {row['co_occurring_issues']}.")
    st.success(f"Suggested management action: {row['suggested_action']}.")

# ---------- 6. Reporting ----------
with tabs[5]:
    scores_r = priority_scores(df, rules_default)
    lines = [
        "Facility Issue Summary Report (PROTOTYPE - synthetic data)",
        f"Records analyzed: {len(df)}",
        f"Date range: {df['date_reported'].min().date()} to {df['date_reported'].max().date()}",
        f"Top category: {df['issue_category'].value_counts().idxmax()}",
        f"High-priority reports: {int((df['priority'] == 'High').sum())}",
        f"Unresolved reports: {int((df['status'] != 'Resolved').sum())}",
        "",
        "Top 5 rooms for maintenance attention:",
    ]
    for _, r in scores_r.head(5).iterrows():
        lines.append(f"  {int(r['rank'])}. {r['room']} ({r['building']}) - score {r['priority_score']}, "
                     f"{r['action']}. {r['suggested_action']}")
    if not rules_default.empty:
        lines += ["", "Strongest association rules:"]
        for _, r in rules_default.head(3).iterrows():
            lines.append(f"  {r['If issue']} -> {r['Then also']} "
                         f"(support {r['Support']:.2f}, confidence {r['Confidence']:.2f}, lift {r['Lift']:.2f})")
    summary = "\n".join(lines) + "\n"
    st.text(summary)
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("Priority list (CSV)", scores_r.to_csv(index=False), "priority_list.csv", "text/csv")
    d2.download_button("Filtered records (CSV)", df.to_csv(index=False), "filtered_reports.csv", "text/csv")
    d3.download_button("Association rules (CSV)", rules_default.to_csv(index=False),
                       "association_rules.csv", "text/csv")
    d4.download_button("Summary (TXT)", summary, "summary_report.txt")
