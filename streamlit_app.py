import streamlit as st
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="Carrier Suggestions", layout="wide")

st.title("Last Mile Carrier Suggestions (Out-of-Network)")
st.markdown(
    "Upload your master (p44) and customer carrier CSVs, choose how many suggestions and metric priority, "
    "and download better-performing out-of-network carriers."
)

# --- Step 1: Upload ---
st.header("1. Upload files")
master_file = st.file_uploader("Upload MASTER dataset (all p44 carriers)", type=["csv"])
customer_file = st.file_uploader("Upload CUSTOMER carrier dataset (carriers customer already uses)", type=["csv"])

# --- Step 2: Parameters ---
st.header("2. Parameters")
num_suggestions = st.number_input("How many top suggestions do you want?", min_value=1, value=10, step=1)

metric_options = [
    ("Data Availability", "higher"),
    ("Milestone Completeness", "higher"),
    ("Scheduled Milestone Completeness", "higher"),
    ("Out for Delivery Milestone Completeness", "higher"),
    ("In Transit Milestone Completeness", "higher"),
    ("Delivered Milestone Completeness", "higher"),
    ("Latency under 1 hr", "lower"),
    ("Latency under 2 hr", "lower"),
    ("Latency bw 1-3 hrs", "lower"),
    ("Latency bw 3-8 hrs", "lower"),
    ("Latency bw 8-24hrs", "lower"),
    ("Latency bw 24-72hrs", "lower"),
    ("Latency over 72hrs", "lower"),
    ("Avg Latency Mins", "lower"),
]

metric_labels = [f"{i+1}. {name} ({direction})" for i, (name, direction) in enumerate(metric_options)]
st.markdown("**Select metric priority order.** Enter numbers in comma-separated order (e.g., `1,2,14`):")
st.text("\n".join(metric_labels))
order_input = st.text_input("Metric priority order", value="1,2,14")
selected_indices = []
try:
    selected_indices = [int(x.strip()) - 1 for x in order_input.split(",") if x.strip()]
except:
    st.error("Invalid metric order input; use numbers like 1,2,14")
selected_metrics = [metric_options[i] for i in selected_indices if 0 <= i < len(metric_options)]

# --- Helper functions ---
def clean_percent(val):
    if isinstance(val, str):
        return float(val.replace("%", "").replace(",", "").strip())
    try:
        return float(val)
    except:
        return pd.NA

def prep(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    if 'Carrier Name' in df.columns:
        df['Carrier Name'] = df['Carrier Name'].str.strip()
    if 'Volume' in df.columns:
        df['Volume'] = (
            df['Volume'].astype(str).str.replace(",", "", regex=False)
            .str.extract(r"(\d+\.?\d*)")[0].astype(float)
        )
    pct_cols = [
        "Data Availability", "Milestone Completeness", "Scheduled Milestone Completeness",
        "Out for Delivery Milestone Completeness", "In Transit Milestone Completeness",
        "Delivered Milestone Completeness", "Latency under 1 hr", "Latency under 2 hr",
        "Latency bw 1-3 hrs", "Latency bw 3-8 hrs", "Latency bw 8-24hrs",
        "Latency bw 24-72hrs", "Latency over 72hrs"
    ]
    for c in pct_cols:
        if c in df.columns:
            df[c] = df[c].apply(clean_percent)
    if "Avg Latency Mins" in df.columns:
        df["Avg Latency Mins"] = pd.to_numeric(df["Avg Latency Mins"], errors="coerce")
    return df

def make_output(master_df, customer_df, metrics, suggestion_count):
    unused = master_df[~master_df["Carrier Name"].str.lower().isin(customer_df["Carrier Name"].str.lower())].copy()
    suggested = []

    for _, sugg in unused.iterrows():
        passes = True
        carrier_to_metrics = {}

        for col, direction in metrics:
            if col not in customer_df.columns or pd.isna(sugg[col]):
                passes = False
                break

            cust_vals = customer_df[col]
            sugg_val = sugg[col]

            if direction == "higher":
                mask = sugg_val >= cust_vals
            else:
                mask = sugg_val <= cust_vals

            if mask.any():
                for ci in customer_df[mask].index:
                    cust_name = customer_df.loc[ci, "Carrier Name"]
                    cust_val = customer_df.loc[ci, col]

                    if pd.isna(cust_val) or pd.isna(sugg_val):
                        continue

                    if abs(sugg_val - cust_val) < 1e-6:
                        tag = "(E)"
                    else:
                        is_better = (sugg_val > cust_val) if direction == "higher" else (sugg_val < cust_val)
                        tag = "(B)" if is_better else "(E)"  # fallback to (E)

                    metric_label = f"{col} {tag}"
                    carrier_to_metrics.setdefault(cust_name, []).append(metric_label)
            else:
                passes = False
                break

        if passes and carrier_to_metrics:
            row = sugg.to_dict()
            i = 1
            for cust_carrier, labeled_metrics in carrier_to_metrics.items():
                row[f"Carrier {i}"] = cust_carrier.title()
                row[f"Reason {i}"] = ", ".join(labeled_metrics)
                i += 1
            suggested.append(row)

    if not suggested:
        return pd.DataFrame()

    df = pd.DataFrame(suggested)
    df = df.head(suggestion_count)
    df.insert(0, "SL No", range(1, len(df)+1))
    for col in df.columns:
        if col.lower().startswith("carrier"):
            df[col] = df[col].astype(str).str.title()
    return df

# --- Step 7: Run logic if both files present ---
if master_file and customer_file and selected_metrics:
    master_df = prep(pd.read_csv(master_file))
    customer_df = prep(pd.read_csv(customer_file))

    output_df = make_output(master_df, customer_df, selected_metrics, num_suggestions)

    st.header("Results Preview")
    if output_df.empty:
        st.warning("No suggestions passed the cascading criteria with the selected metric order.")
    else:
        st.dataframe(output_df)

        # Legend
        legend_rows = [
            ["Column", "Explanation"],
            ["Carrier N", "Customer carrier that this suggestion outperformed on one or more metrics"],
            ["Reason N", "Metrics (in order) where the suggested carrier outperformed that customer carrier. (B)=Better, (E)=Equal"]
        ]
        legend_df = pd.DataFrame(legend_rows[1:], columns=legend_rows[0])
        st.subheader("Legend")
        st.table(legend_df)

        # Download
        def to_excel_bytes(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Suggestions")
                legend_df.to_excel(writer, index=False, sheet_name="Legend")
                writer.save()
            return output.getvalue()

        if st.button("Download Suggestions"):
            if not output_df.empty:
                data = to_excel_bytes(output_df)
                st.download_button(
                    "Click to download Excel",
                    data,
                    file_name=f"Top_{num_suggestions}_Suggestions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
