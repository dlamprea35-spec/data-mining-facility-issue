"""
Generates a SYNTHETIC sample dataset of classroom facility issue reports.
FOR PROTOTYPING ONLY - not real ADSSU records. Replace with the anonymized
records approved by General Services before Capstone 2 analysis.
Building, college and room names are generic placeholders.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

BUILDINGS = {
    "Building A": "College of Education",
    "Building B": "College of Computing",
    "Building C": "College of Engineering",
    "Building D": "College of Agriculture",
    "Building E": "College of Business",
}
ROOMS_PER_BUILDING = 8  # rooms 101-108

TEMPLATES = {
    "Electrical": [
        "walay kuryente ang outlet sa {room}",
        "sira ang ilaw sa {room}, dili mosiga",
        "busted light bulb in {room}",
        "nag-spark ang switch sa {room}",
        "ang outlet sa {room} dili mugana",
        "flickering lights sa {room}",
    ],
    "Ventilation": [
        "sira ang electric fan sa {room}",
        "dili mulupad ang electric fan sa {room}",
        "broken ceiling fan in {room}, init kaayo",
        "ang aircon sa {room} dili cold",
        "electric fan sa {room} naghimo og kasaba",
        "init kaayo sa {room}, walay hangin",
    ],
    "Furniture": [
        "naguba ang armchair sa {room}",
        "broken chair in {room}",
        "sira ang lamesa sa teacher sa {room}",
        "nahulog ang whiteboard sa {room}",
        "kulang ang armchairs sa {room}",
        "loose desk legs sa {room}",
    ],
    "Audio-Visual": [
        "dili mugana ang projector sa {room}",
        "walay display ang projector sa {room}",
        "projector in {room} no signal",
        "sira ang speaker sa {room}",
        "ang HDMI cable sa {room} naputol",
        "blurry ang projector sa {room}",
    ],
    "Plumbing": [
        "nag-agas ang gripo sa CR duol sa {room}",
        "clogged sink near {room}",
        "baha ang CR sa {room}",
        "sira ang flush sa CR sa {room}",
        "leaking faucet near {room}",
    ],
    "Structural": [
        "nag-tulo ang atop sa {room}",
        "roof leak in {room} when it rains",
        "gi-crack ang dingding sa {room}",
        "sira ang pultahan sa {room}",
        "broken window glass in {room}",
        "dili ma-lock ang pultahan sa {room}",
    ],
    "Other": [
        "hugaw kaayo ang {room}",
        "naay ipis sa {room}",
        "walay basurahan sa {room}",
        "problema sa {room}, palihug check",
    ],
}
VAGUE = ["sira", "problema", "palihug check", "kinahanglan i-repair"]

BASE_P = {"Electrical": .060, "Ventilation": .070, "Furniture": .055,
          "Audio-Visual": .045, "Plumbing": .035, "Structural": .030, "Other": .030}
# month-of-year effects (rainy season boosts plumbing/structural)
RAINY = {11: 1.6, 12: 2.0, 1: 2.0, 2: 1.6}
# co-occurrence boosts within the same room and month (planted pattern for the demo)
BOOST = {("Electrical", "Ventilation"): 5.0, ("Electrical", "Audio-Visual"): 4.0,
         ("Plumbing", "Structural"): 4.0, ("Furniture", "Other"): 2.5}

rooms = [(b, f"{b.split()[1]}-{100 + r}") for b in BUILDINGS for r in range(1, ROOMS_PER_BUILDING + 1)]
# some rooms are "problem-prone"
prone = {room: rng.uniform(0.6, 2.4) for _, room in rooms}

months = pd.period_range("2023-08", "2026-07", freq="M")
rows = []
for m in months:
    for b, room in rooms:
        mult = prone[room]
        present = set()
        for cat, p in BASE_P.items():
            pp = p * mult
            if cat in ("Plumbing", "Structural"):
                pp *= RAINY.get(m.month, 1.0)
            if rng.random() < min(pp, 0.6):
                present.add(cat)
        for (a, c), k in BOOST.items():
            if a in present and c not in present and rng.random() < min(BASE_P[c] * mult * k, 0.7):
                present.add(c)
        for cat in present:
            day = int(rng.integers(1, 28))
            date = pd.Timestamp(year=m.year, month=m.month, day=day)
            text = str(rng.choice(TEMPLATES[cat])).format(room=room)
            label = cat
            if rng.random() < 0.08:  # vague / noisy description
                text = f"{rng.choice(VAGUE)} sa {room}"
            if rng.random() < 0.04:  # inconsistent label noise
                label = str(rng.choice(list(TEMPLATES)))
            high_kw = any(k in text for k in ["spark", "baha", "tulo", "leak", "crack"])
            pr = rng.choice(["High", "Medium", "Low"],
                            p=[.55, .35, .10] if high_kw else
                            [.15, .50, .35] if cat in ("Electrical", "Structural", "Plumbing") else
                            [.08, .40, .52])
            status = rng.choice(["Resolved", "In Progress", "Pending"], p=[.72, .15, .13])
            days = int(rng.integers(1, 10 if pr == "High" else 25)) if status == "Resolved" else None
            rows.append({
                "date_reported": date.date().isoformat(),
                "college": BUILDINGS[b],
                "building": b,
                "room": room,
                "issue_description": text,
                "issue_category": label,
                "priority": pr,
                "status": status,
                "date_resolved": (date + pd.Timedelta(days=days)).date().isoformat() if days else "",
            })

df = pd.DataFrame(rows).sort_values("date_reported").reset_index(drop=True)
df.insert(0, "report_id", [f"SYN-{i + 1:04d}" for i in range(len(df))])
df.to_csv("issue_reports_synthetic.csv", index=False)
print(f"Generated {len(df)} synthetic records -> issue_reports_synthetic.csv")
