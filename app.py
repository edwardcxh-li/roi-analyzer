import os, io, duckdb, pandas as pd
import streamlit as st
import altair as alt
from dotenv import load_dotenv
from openai import OpenAI

# ---------- env ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
client = OpenAI()

st.set_page_config(page_title="Marketing Mix & ROI Analyzer", layout="wide")
st.title("📈 Marketing Mix & ROI Analyzer")
st.caption("CSV + DuckDB (SQL) + Pandas + Streamlit + GPT insights")

# ---------- data loading ----------
SAMPLE_PATH = "data/marketing_data.csv"

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    for c in ["spend","impressions","clicks","conversions","revenue"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

@st.cache_data
def load_uploaded_csv(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content), parse_dates=["date"])
    for c in ["spend","impressions","clicks","conversions","revenue"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

# ---------- sidebar: source & filters ----------
with st.sidebar:
    st.header("Data")
    upl = st.file_uploader("Upload CSV (date,channel,spend,impressions,clicks,conversions,revenue)", type=["csv"])
    use_sample = st.checkbox("Use bundled sample data", value=(upl is None))
    st.markdown("---")

# pick data source
if upl is not None and not use_sample:
    df = load_uploaded_csv(upl.read())
else:
    if not os.path.exists(SAMPLE_PATH):
        st.error(f"Sample not found: {SAMPLE_PATH}")
        st.stop()
    df = load_csv(SAMPLE_PATH)

# ---------- SQL helper ----------
def run_sql(q: str) -> pd.DataFrame:
    return duckdb.query(q).to_df()

# ---------- filters ----------
with st.sidebar:
    st.header("Filters")
    channels = sorted(df["channel"].unique().tolist())
    pick = st.multiselect("Channels", channels, default=channels)
    start, end = df["date"].min(), df["date"].max()
    date_range = st.date_input("Date range", (start, end), min_value=start, max_value=end)
    st.markdown("---")

flt = df.copy()
if pick:
    flt = flt[flt["channel"].isin(pick)]
if isinstance(date_range, tuple) and len(date_range) == 2:
    flt = flt[(flt["date"] >= pd.Timestamp(date_range[0])) & (flt["date"] <= pd.Timestamp(date_range[1]))]

# register filtered view
duckdb.sql("CREATE OR REPLACE VIEW base AS SELECT * FROM flt")

# ---------- what-if controls ----------
with st.sidebar:
    st.header("What-if: Budget Adjustment")
    st.caption("Apply +/- % to *spend* by channel, then recompute metrics.")
    adj = {}
    for ch in channels:
        adj[ch] = st.slider(f"{ch} %", min_value=-50, max_value=50, value=0, step=5, format="%d%%")
    apply_adj = st.checkbox("Apply what-if adjustments", value=False)

def apply_spend_adjustments(frame: pd.DataFrame, adjustments: dict[str,int]) -> pd.DataFrame:
    if not apply_adj or frame.empty:
        return frame
    g = frame.copy()
    factors = {k: 1 + (v/100.0) for k, v in adjustments.items()}
    g["spend"] = g.apply(lambda r: r["spend"] * factors.get(r["channel"], 1.0), axis=1)
    return g

sim = apply_spend_adjustments(flt, adj)
duckdb.sql("CREATE OR REPLACE VIEW marketing AS SELECT * FROM sim")

# ---------- SQL templates ----------
SQL_BY_CHANNEL = """
SELECT channel,
       SUM(spend) AS spend,
       SUM(revenue) AS revenue,
       SUM(clicks) AS clicks,
       SUM(conversions) AS conversions,
       CASE WHEN SUM(spend)=0 THEN NULL ELSE (SUM(revenue)-SUM(spend))/SUM(spend) END AS roi,
       CASE WHEN SUM(clicks)=0 THEN NULL ELSE SUM(spend)/SUM(clicks) END AS cpc,
       CASE WHEN SUM(conversions)=0 THEN NULL ELSE SUM(spend)/SUM(conversions) END AS cac,
       CASE WHEN SUM(impressions)=0 THEN NULL ELSE CAST(SUM(clicks) AS DOUBLE) / SUM(impressions) END AS ctr
FROM marketing
GROUP BY channel
ORDER BY roi DESC NULLS LAST;
"""

SQL_BY_DAY = """
SELECT date::DATE AS date,
       SUM(spend) AS spend,
       SUM(revenue) AS revenue,
       SUM(clicks) AS clicks,
       SUM(conversions) AS conversions
FROM marketing
GROUP BY 1
ORDER BY 1;
"""

# ---------- top KPIs ----------
kpi = run_sql("""
SELECT
  SUM(spend) AS spend,
  SUM(revenue) AS revenue,
  CASE WHEN SUM(conversions)=0 THEN NULL ELSE SUM(spend)/SUM(conversions) END AS cac,
  CASE WHEN SUM(spend)=0 THEN NULL ELSE (SUM(revenue)-SUM(spend))/SUM(spend) END AS roi
FROM marketing;
""")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Spend", f"${kpi['spend'][0]:,.0f}")
c2.metric("Total Revenue", f"${kpi['revenue'][0]:,.0f}")
c3.metric("CAC", "—" if pd.isna(kpi['cac'][0]) else f"${kpi['cac'][0]:,.2f}")
c4.metric("ROI", "—" if pd.isna(kpi['roi'][0]) else f"{kpi['roi'][0]*100:,.1f}%")

# ---------- Channel Performance ----------
st.subheader("Channel Performance")
by_channel = run_sql(SQL_BY_CHANNEL)
st.dataframe(by_channel, use_container_width=True)

# downloads
col_a, col_b, col_c = st.columns(3)
col_a.download_button("⬇️ Download raw (filtered)", flt.to_csv(index=False), "raw_filtered.csv")
col_b.download_button("⬇️ Download what-if (simulated)", sim.to_csv(index=False), "raw_simulated.csv")
col_c.download_button("⬇️ Download channel summary", by_channel.to_csv(index=False), "channel_summary.csv")

# chart ROI by channel
if not by_channel.empty:
    chart = alt.Chart(by_channel).mark_bar().encode(
        x=alt.X("channel:N", sort='-y', title="Channel"),
        y=alt.Y("roi:Q", title="ROI"),
        tooltip=list(by_channel.columns)
    )
    st.altair_chart(chart, use_container_width=True)

# ---------- Daily Trends ----------
st.subheader("Daily Trends")
by_day = run_sql(SQL_BY_DAY)
st.dataframe(by_day, use_container_width=True)

if not by_day.empty:
    line = alt.Chart(by_day).mark_line().encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("revenue:Q", title="Revenue"),
        tooltip=list(by_day.columns)
    ).properties(title="Revenue by Day")
    st.altair_chart(line, use_container_width=True)

# ---------- AI recommendations (cached) ----------
@st.cache_data(show_spinner=False)
def cached_ai_reco(model: str, csv_snippet: str, applied: bool) -> str:
    if client is None:
        return "No OPENAI_API_KEY set. Skipping AI suggestions."
    prompt = (
        "You are a performance marketing analyst. Given this channel summary (CSV with spend, revenue, ROI, CPC, CAC, CTR), "
        "write 3–5 specific, actionable recommendations (short bullets) for budget reallocation to improve overall ROI. "
        "Be concrete: mention channels to decrease/increase and why; call out CAC/CTR issues if any. Keep under 120 words.\n\n"
        f"{csv_snippet}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":"Be concise, data-driven, and business-friendly."},
                {"role":"user","content":prompt}
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(AI suggestion unavailable: {e})"

st.subheader("AI Budget Recommendations")
csv_summary = by_channel.to_csv(index=False)
note = " (with what-if adjustments applied)" if apply_adj else ""
with st.spinner("Analyzing performance..."):
    suggestion = cached_ai_reco(MODEL, csv_summary, apply_adj)
st.info(f"**Recommendations{note}:**\n\n{suggestion}")
