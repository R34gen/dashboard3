from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Project 1 | Rekognisi Mata Kuliah",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLE
# ============================================================
st.markdown("""
<style>
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1400px;
}
h1, h2, h3 { letter-spacing: -0.04em; }
.subtitle {
    color: #9ca3af;
    font-size: 1.05rem;
    margin-top: -0.5rem;
    margin-bottom: 1rem;
}
.kpi-card {
    background: linear-gradient(135deg, #ffffff 0%, #f5f7fb 100%);
    color: #111827;
    border-radius: 18px;
    padding: 22px 24px;
    border: 1px solid rgba(255,255,255,0.08);
    min-height: 118px;
    box-shadow: 0 16px 40px rgba(0,0,0,0.18);
}
.kpi-label {
    color: #6b7280;
    font-size: 0.95rem;
    font-weight: 650;
    margin-bottom: 12px;
}
.kpi-value {
    color: #111827;
    font-size: 2.0rem;
    font-weight: 850;
    line-height: 1.15;
}
.insight-box {
    background-color: #1f2937;
    color: #f9fafb;
    padding: 18px 22px;
    border-radius: 14px;
    border-left: 6px solid #ff4b4b;
    margin-top: 14px;
    margin-bottom: 16px;
    line-height: 1.7;
}
.success-box {
    background-color: #102a1f;
    color: #d1fae5;
    padding: 18px 22px;
    border-radius: 14px;
    border-left: 6px solid #22c55e;
    margin-top: 14px;
    margin-bottom: 16px;
    line-height: 1.7;
}
.warning-box {
    background-color: #2a1f10;
    color: #ffedd5;
    padding: 18px 22px;
    border-radius: 14px;
    border-left: 6px solid #f97316;
    margin-top: 14px;
    margin-bottom: 16px;
    line-height: 1.7;
}
.gray-box {
    background-color: #111827;
    color: #e5e7eb;
    padding: 18px 22px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.10);
    margin-top: 14px;
    margin-bottom: 16px;
    line-height: 1.7;
}
.section-divider {
    margin-top: 1.5rem;
    margin-bottom: 1.8rem;
    border: none;
    height: 1px;
    background: rgba(255,255,255,0.10);
}
div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 16px;
    border-bottom: 1px solid rgba(255,255,255,0.10);
}
.stTabs [data-baseweb="tab"] {
    height: 48px;
    white-space: pre-wrap;
    border-radius: 0;
    padding-left: 0;
    padding-right: 0;
    font-weight: 700;
}
.stTabs [aria-selected="true"] {
    color: #ff4b4b;
    border-bottom: 3px solid #ff4b4b;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
SEGMENT_ORDER = [
    "Champions", "Loyal Customers", "Big Spenders",
    "High Value At Risk", "At Risk", "Regular Customers", "Lost Customers"
]
PRIORITY_ORDER = ["Priority 1", "Priority 2", "Priority 3", "Priority 4", "Priority 5", "Priority 6"]
NON_PRODUCT = [
    "POSTAGE", "DOTCOM POSTAGE", "CARRIAGE", "MANUAL", "BANK CHARGES",
    "AMAZON FEE", "SAMPLES", "ADJUST", "ADJUSTMENT", "CRUK COMMISSION"
]
PLOTLY_TEMPLATE = "plotly_dark"
PRIMARY_COLOR = "#7cc4ff"

# ============================================================
# HELPERS
# ============================================================
def money_fmt(x):
    try: return f"{float(x):,.2f}"
    except Exception: return "-"

def int_fmt(x):
    try: return f"{int(x):,}"
    except Exception: return "-"

def kpi_card(label, value):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def section_note(kind, html):
    cls = {
        "info": "insight-box",
        "success": "success-box",
        "warning": "warning-box",
        "gray": "gray-box",
    }.get(kind, "insight-box")
    st.markdown(f'<div class="{cls}">{html}</div>', unsafe_allow_html=True)

def display_df(df, height=None):
    if height is None:
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True, height=height)

@st.cache_data(show_spinner=False)
def load_csv(filename, data_dir):
    path = Path(data_dir) / filename
    if not path.exists():
        return None
    return pd.read_csv(path)

def clean_text_columns(df):
    if df is None:
        return None
    out = df.copy()
    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].astype(str).str.strip()
    return out

def safe_numeric(df, cols):
    if df is None:
        return df
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def rename_for_display(df):
    mapping = {
        "CustomerID": "Customer ID",
        "RFM_Segment": "RFM Segment",
        "Cluster_Label": "Cluster Label",
        "Business_Priority": "Business Priority",
        "Recommended_Action": "Recommended Action",
        "Risk_Score": "Risk Score",
        "Value_Score": "Value Score",
        "Business_Score": "Business Score",
        "Avg_Recency": "Avg Recency",
        "Avg_Frequency": "Avg Frequency",
        "Avg_Monetary": "Avg Monetary",
        "Avg_Business_Score": "Avg Business Score",
        "Total_Monetary": "Total Monetary",
        "Total_Revenue": "Total Revenue",
        "Total_Transaction": "Total Transaction",
        "Total_Customer": "Total Customer",
        "ProductName": "Product Name",
        "ProductKey": "Product Key",
        "CountryKey": "Country Key",
        "DateKey": "Date Key",
        "MonthYear": "Month Year",
        "Mata_Kuliah": "Mata Kuliah",
        "Indikator_MBKM": "Indikator MBKM",
        "Bukti_Project_1": "Bukti Project 1",
        "Output_File": "Output File",
        "Catatan_Kritis": "Catatan Kritis",
    }
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})

def make_bar(df, x, y, title, order=None):
    fig = px.bar(df, x=x, y=y, title=title, template=PLOTLY_TEMPLATE, color_discrete_sequence=[PRIMARY_COLOR])
    if order:
        fig.update_xaxes(categoryorder="array", categoryarray=order)
    fig.update_layout(
        title_font_size=18, height=420, margin=dict(l=20, r=20, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def make_hbar(df, x, y, title):
    fig = px.bar(df, x=x, y=y, orientation="h", title=title, template=PLOTLY_TEMPLATE, color_discrete_sequence=[PRIMARY_COLOR])
    fig.update_layout(
        title_font_size=18, height=460, margin=dict(l=20, r=20, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed")
    )
    return fig

def prepare_sales_model(fact_sales, dim_customer, dim_product, dim_country, dim_date):
    sales = fact_sales.copy()
    if dim_product is not None and {"ProductKey", "ProductName"}.issubset(dim_product.columns):
        sales = sales.merge(dim_product[["ProductKey", "ProductName"]], on="ProductKey", how="left")
    if dim_country is not None and {"CountryKey", "Country"}.issubset(dim_country.columns):
        sales = sales.merge(dim_country[["CountryKey", "Country"]], on="CountryKey", how="left")
    if dim_date is not None and {"DateKey"}.issubset(dim_date.columns):
        date_cols = [c for c in ["DateKey", "Date", "Year", "Month", "MonthName", "Quarter", "MonthYear"] if c in dim_date.columns]
        sales = sales.merge(dim_date[date_cols], on="DateKey", how="left")
    if dim_customer is not None and "CustomerID" in dim_customer.columns:
        cust_cols = [c for c in [
            "CustomerID", "RFM_Segment", "Cluster_Label", "Business_Priority",
            "Recommended_Action", "Business_Score"
        ] if c in dim_customer.columns]
        sales = sales.merge(dim_customer[cust_cols], on="CustomerID", how="left")
    return sales

def remove_non_product(df, product_col):
    out = df.copy()
    if product_col in out.columns:
        out = out[~out[product_col].astype(str).str.upper().str.strip().isin(NON_PRODUCT)]
    return out

def course_indicator_table(course):
    data = {
        "Softskill / Kepemimpinan": [
            ["Analisis masalah, keputusan, dan risiko", "method_decision.csv, limitations, alasan scope, keputusan tidak menghapus outlier valid"],
            ["Komunikasi dan koordinasi", "logbook, catatan konsultasi, revisi berdasarkan masukan mentor/dosen"],
            ["Tanggung jawab dokumentasi", "notebook, output CSV, dashboard, screenshot, laporan BAB 4"],
            ["Catatan kritis", "Dashboard hanya bukti pendukung; softskill tetap perlu logbook dan bukti proses"],
        ],
        "Kecerdasan Bisnis": [
            ["Konsep BI, ETL, reporting, visualisasi", "ETL workflow, KPI, Sales Performance, dashboard Streamlit"],
            ["Star schema / model data", "fact_sales, dim_customer, dim_product, dim_country, dim_date"],
            ["Tools BI / reporting", "Python, Pandas, Plotly, Streamlit"],
            ["Workflow BI", "Extract → Transform → Model → Load → Visualize → Decision"],
        ],
        "Analisis Segmentasi Pelanggan": [
            ["Konsep dan tipe segmentasi", "Segmentasi perilaku transaksi berbasis RFM; geografis sederhana berdasarkan Country"],
            ["Pemilihan metode segmentasi", "RFM Analysis sebagai metode utama; Agglomerative Clustering sebagai pendukung"],
            ["Tools segmentasi", "Python, Pandas, Scikit-learn, Streamlit"],
            ["Workflow segmentasi", "cleaning → RFM → scoring → segment → cluster validation → strategi"],
        ],
        "Pemodelan Data Bisnis": [
            ["Konsep return, value, dan risiko bisnis", "Monetary sebagai nilai pelanggan; Recency sebagai proxy risiko pasif"],
            ["Pendekatan pemodelan keputusan", "rule-based business decision model berbasis Risk/Value/Business Score"],
            ["Tools pemodelan", "Python, Pandas, NumPy, Streamlit"],
            ["Workflow pemodelan", "RFM Segment → Risk Score → Value Score → Business Score → Priority → Action"],
        ],
        "Intuisi dan Wawasan Data": [
            ["Memahami data dunia nyata", "Online Retail transaction behavior, outlier, negara dominan, keterbatasan dataset"],
            ["Menyusun insight", "KPI, tren revenue, top country, top product, segmentasi pelanggan"],
            ["Keputusan berbasis data", "retensi Champions, reaktivasi High Value At Risk, win-back hemat biaya"],
            ["Wawasan aplikatif", "Executive Summary, Business Recommendation, Limitations"],
        ],
    }
    return pd.DataFrame(data[course], columns=["Indikator yang Dibuktikan", "Bukti Project 1"])

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("BAB 4 Rekognisi")
st.sidebar.caption("Project 1 — Customer Intelligence Online Retail")

data_dir = st.sidebar.text_input(
    "Data folder",
    value="project1_output",
    help="Folder berisi hasil export final dari Colab.",
)

page = st.sidebar.radio(
    "Navigasi berdasarkan mata kuliah",
    [
        "Ringkasan Project 1",
        "4.1 Softskill / Kepemimpinan",
        "4.2 Kecerdasan Bisnis",
        "4.3 Analisis Segmentasi Pelanggan",
        "4.4 Pemodelan Data Bisnis",
        "4.5 Intuisi dan Wawasan Data",
        "Lampiran Data Model",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filter")
st.sidebar.caption("Kosongkan filter untuk menampilkan semua data.")

# ============================================================
# LOAD DATA
# ============================================================
fact_sales = clean_text_columns(load_csv("fact_sales.csv", data_dir))
dim_customer = clean_text_columns(load_csv("dim_customer.csv", data_dir))
dim_product = clean_text_columns(load_csv("dim_product.csv", data_dir))
dim_country = clean_text_columns(load_csv("dim_country.csv", data_dir))
dim_date = clean_text_columns(load_csv("dim_date.csv", data_dir))

business_summary = clean_text_columns(load_csv("business_summary.csv", data_dir))
rfm_cluster_matrix = clean_text_columns(load_csv("rfm_cluster_matrix.csv", data_dir))
bi_workflow = clean_text_columns(load_csv("bi_workflow.csv", data_dir))
segmentation_workflow = clean_text_columns(load_csv("segmentation_workflow.csv", data_dir))
business_modeling_workflow = clean_text_columns(load_csv("business_modeling_workflow.csv", data_dir))
method_decision = clean_text_columns(load_csv("method_decision.csv", data_dir))
course_mapping = clean_text_columns(load_csv("course_mapping_project1.csv", data_dir))

if fact_sales is None or dim_customer is None:
    st.error("File utama belum ditemukan. Pastikan folder project1_output berisi fact_sales.csv dan dim_customer.csv.")
    st.stop()

fact_sales = safe_numeric(fact_sales, ["Revenue", "Quantity", "UnitPrice", "DateKey", "ProductKey", "CountryKey"])
dim_customer = safe_numeric(dim_customer, ["Monetary", "Recency", "Frequency", "Risk_Score", "Value_Score", "Business_Score"])
if dim_date is not None:
    dim_date = safe_numeric(dim_date, ["Year", "Month", "DateKey"])
    if "Date" in dim_date.columns:
        dim_date["Date"] = pd.to_datetime(dim_date["Date"], errors="coerce")

sales_model = prepare_sales_model(fact_sales, dim_customer, dim_product, dim_country, dim_date)

years = sorted([int(y) for y in sales_model["Year"].dropna().unique()]) if "Year" in sales_model.columns else []
countries = sorted(sales_model["Country"].dropna().unique()) if "Country" in sales_model.columns else []
segments = [s for s in SEGMENT_ORDER if "RFM_Segment" in dim_customer.columns and s in dim_customer["RFM_Segment"].dropna().unique()]
priorities = [p for p in PRIORITY_ORDER if "Business_Priority" in dim_customer.columns and p in dim_customer["Business_Priority"].dropna().unique()]

selected_year = st.sidebar.multiselect("Year", years, default=[])
selected_country = st.sidebar.multiselect("Country", countries, default=[])
selected_segment = st.sidebar.multiselect("RFM Segment", segments, default=[])
selected_priority = st.sidebar.multiselect("Business Priority", priorities, default=[])

filtered_customer = dim_customer.copy()
if selected_segment and "RFM_Segment" in filtered_customer.columns:
    filtered_customer = filtered_customer[filtered_customer["RFM_Segment"].isin(selected_segment)]
if selected_priority and "Business_Priority" in filtered_customer.columns:
    filtered_customer = filtered_customer[filtered_customer["Business_Priority"].isin(selected_priority)]

filtered_sales = sales_model.copy()
if "CustomerID" in filtered_customer.columns and "CustomerID" in filtered_sales.columns:
    filtered_sales = filtered_sales[filtered_sales["CustomerID"].astype(str).isin(set(filtered_customer["CustomerID"].astype(str)))]
if selected_year and "Year" in filtered_sales.columns:
    filtered_sales = filtered_sales[filtered_sales["Year"].isin(selected_year)]
if selected_country and "Country" in filtered_sales.columns:
    filtered_sales = filtered_sales[filtered_sales["Country"].isin(selected_country)]

active_filters = []
if selected_year: active_filters.append(f"Year: {', '.join(map(str, selected_year))}")
if selected_country: active_filters.append(f"Country: {len(selected_country)} selected")
if selected_segment: active_filters.append(f"Segment: {len(selected_segment)} selected")
if selected_priority: active_filters.append(f"Priority: {len(selected_priority)} selected")
filter_status = "All data" if not active_filters else " | ".join(active_filters)

st.sidebar.markdown("---")
st.sidebar.subheader("Download")
st.sidebar.download_button(
    "Customer Segment CSV",
    data=dim_customer.to_csv(index=False).encode("utf-8"),
    file_name="rfm_customer_segment.csv",
    mime="text/csv",
)
if business_summary is not None:
    st.sidebar.download_button(
        "Business Summary CSV",
        data=business_summary.to_csv(index=False).encode("utf-8"),
        file_name="business_summary.csv",
        mime="text/csv",
    )

# Common KPI
total_revenue = filtered_sales["Revenue"].sum() if "Revenue" in filtered_sales.columns else 0
total_transaction = filtered_sales["InvoiceNo"].nunique() if "InvoiceNo" in filtered_sales.columns else 0
total_customer = filtered_sales["CustomerID"].nunique() if "CustomerID" in filtered_sales.columns else 0
total_product = filtered_sales["ProductKey"].nunique() if "ProductKey" in filtered_sales.columns else 0
aov = total_revenue / total_transaction if total_transaction else 0

# ============================================================
# PAGES
# ============================================================
if page == "Ringkasan Project 1":
    st.title("📊 Ringkasan Project 1")
    st.markdown('<div class="subtitle">Customer Intelligence Dashboard untuk data transaksi Online Retail.</div>', unsafe_allow_html=True)
    st.caption(f"Filter aktif: {filter_status}")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("Total Revenue", money_fmt(total_revenue))
    with c2: kpi_card("Transactions", int_fmt(total_transaction))
    with c3: kpi_card("Customers", int_fmt(total_customer))
    with c4: kpi_card("Products", int_fmt(total_product))
    with c5: kpi_card("AOV", money_fmt(aov))

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Executive Summary")
        section_note("info", """
        <b>Project ini bukan sekadar dashboard.</b><br>
        Project ini membentuk alur customer intelligence dari data transaksi: cleaning, feature engineering,
        KPI, RFM segmentation, clustering pendukung, business priority, recommended action, dan dashboard interaktif.
        <br><br>
        <b>Batas klaim:</b> segmentasi bersifat perilaku transaksi, bukan demografis. Business priority adalah model keputusan berbasis aturan, bukan prediksi churn.
        """)
    with right:
        st.subheader("Mapping BAB 4")
        mapping = pd.DataFrame({
            "Subbab": ["4.1", "4.2", "4.3", "4.4", "4.5"],
            "Mata Kuliah": ["Softskill / Kepemimpinan", "Kecerdasan Bisnis", "Analisis Segmentasi Pelanggan", "Pemodelan Data Bisnis", "Intuisi dan Wawasan Data"],
            "Bukti Utama": ["method decision, logbook, kendala-solusi", "ETL, star schema, KPI, dashboard", "RFM, cluster pendukung, profil segmen", "risk/value/business score, priority, action", "insight, interpretasi, rekomendasi"],
        })
        display_df(mapping)

elif page == "4.1 Softskill / Kepemimpinan":
    st.title("4.1 Softskill / Kepemimpinan")
    st.markdown('<div class="subtitle">Fokus bukti: proses, keputusan, risiko, komunikasi, dan dokumentasi.</div>', unsafe_allow_html=True)

    section_note("warning", """
    <b>Kritik penting:</b> dashboard bukan bukti utama softskill. Softskill harus dibuktikan dengan proses:
    logbook, catatan konsultasi, timeline, kendala-solusi, dan keputusan metodologis.
    Halaman ini hanya merapikan bukti keputusan; jangan jadikan ini satu-satunya bukti Kepemimpinan.
    """)

    st.subheader("Indikator dan Bukti")
    display_df(course_indicator_table("Softskill / Kepemimpinan"))

    st.subheader("Method Decision")
    if method_decision is not None:
        display_df(rename_for_display(method_decision), height=420)
    else:
        st.caption("method_decision.csv tidak ditemukan.")

    st.subheader("Bukti Tambahan yang Harus Masuk Laporan")
    display_df(pd.DataFrame({
        "Bukti": ["Logbook", "Timeline", "Catatan konsultasi", "Screenshot progres", "Kendala dan solusi"],
        "Fungsi": [
            "Membuktikan konsistensi pengerjaan",
            "Membuktikan perencanaan dan manajemen waktu",
            "Membuktikan komunikasi dan revisi berdasarkan masukan",
            "Membuktikan progress nyata",
            "Membuktikan kemampuan menghadapi masalah"
        ]
    }))

elif page == "4.2 Kecerdasan Bisnis":
    st.title("4.2 Kecerdasan Bisnis")
    st.markdown('<div class="subtitle">Fokus bukti: ETL, star schema, KPI, reporting, visualisasi, workflow BI.</div>', unsafe_allow_html=True)
    st.caption(f"Filter aktif: {filter_status}")

    st.subheader("Indikator dan Bukti")
    display_df(course_indicator_table("Kecerdasan Bisnis"))

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Total Revenue", money_fmt(total_revenue))
    with c2: kpi_card("Transactions", int_fmt(total_transaction))
    with c3: kpi_card("Customers", int_fmt(total_customer))
    with c4: kpi_card("AOV", money_fmt(aov))

    if len(filtered_sales) > 0:
        top_country_name = "-"
        if "Country" in filtered_sales.columns and filtered_sales["Country"].notna().any():
            top_country_name = filtered_sales.groupby("Country")["Revenue"].sum().sort_values(ascending=False).index[0]
        section_note("info", f"""
        <b>Ringkasan BI:</b> dashboard menampilkan total revenue <b>{money_fmt(total_revenue)}</b>
        dari <b>{int_fmt(total_transaction)}</b> transaksi. Negara dominan pada filter aktif adalah <b>{top_country_name}</b>.
        Bagian ini menunjukkan reporting dan visualisasi KPI, bukan hanya tabel statis.
        """)

    if {"MonthYear", "Revenue"}.issubset(filtered_sales.columns):
        monthly = (
            filtered_sales.groupby(["Year", "Month", "MonthYear"], as_index=False)
            .agg(Revenue=("Revenue", "sum"))
            .sort_values(["Year", "Month"])
        )
        fig = px.line(monthly, x="MonthYear", y="Revenue", markers=True, title="Monthly Revenue Trend", template=PLOTLY_TEMPLATE, color_discrete_sequence=[PRIMARY_COLOR])
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=40), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if "Country" in filtered_sales.columns:
            country_chart = filtered_sales.groupby("Country", as_index=False).agg(Revenue=("Revenue", "sum")).sort_values("Revenue", ascending=False).head(10)
            st.plotly_chart(make_bar(country_chart, "Country", "Revenue", "Top 10 Country by Revenue"), use_container_width=True)
    with col2:
        if "ProductName" in filtered_sales.columns:
            product_source = remove_non_product(filtered_sales, "ProductName")
            product_chart = (
                product_source.groupby("ProductName", as_index=False)
                .agg(Revenue=("Revenue", "sum"), Transaction=("InvoiceNo", "nunique"))
                .query("Transaction >= 50")
                .sort_values("Revenue", ascending=False)
                .head(10)
            )
            st.plotly_chart(make_hbar(product_chart, "Revenue", "ProductName", "Top 10 Consistent Product by Revenue"), use_container_width=True)

    st.subheader("Star Schema")
    star_schema = pd.DataFrame({
        "Table": ["fact_sales", "dim_customer", "dim_product", "dim_country", "dim_date"],
        "Role": ["Fact table", "Customer dimension", "Product dimension", "Country dimension", "Date dimension"],
        "Key": ["CustomerID, ProductKey, CountryKey, DateKey", "CustomerID", "ProductKey", "CountryKey", "DateKey"],
        "Rows": [len(fact_sales), len(dim_customer), len(dim_product) if dim_product is not None else 0, len(dim_country) if dim_country is not None else 0, len(dim_date) if dim_date is not None else 0],
    })
    display_df(star_schema)

    st.graphviz_chart("""
    digraph {
        graph [rankdir=LR, bgcolor="transparent"]
        node [shape=box, style="rounded,filled", fillcolor="#1f2937", fontcolor="white", color="#ff4b4b"]
        edge [fontcolor="white", color="#9ca3af"]
        dim_customer [label="dim_customer\\nCustomerID"]
        dim_product [label="dim_product\\nProductKey"]
        dim_country [label="dim_country\\nCountryKey"]
        dim_date [label="dim_date\\nDateKey"]
        fact_sales [label="fact_sales\\nCustomerID, ProductKey, CountryKey, DateKey"]
        dim_customer -> fact_sales [label="1 : many"]
        dim_product -> fact_sales [label="1 : many"]
        dim_country -> fact_sales [label="1 : many"]
        dim_date -> fact_sales [label="1 : many"]
    }
    """)

    if bi_workflow is not None:
        st.subheader("BI Workflow")
        display_df(rename_for_display(bi_workflow.replace("Power BI", "Streamlit", regex=True)), height=320)

elif page == "4.3 Analisis Segmentasi Pelanggan":
    st.title("4.3 Analisis Segmentasi Pelanggan")
    st.markdown('<div class="subtitle">Fokus bukti: konsep segmentasi, RFM, clustering pendukung, workflow, dan strategi segmen.</div>', unsafe_allow_html=True)
    st.caption(f"Filter aktif: {filter_status}")

    st.subheader("Indikator dan Bukti")
    display_df(course_indicator_table("Analisis Segmentasi Pelanggan"))

    section_note("warning", """
    <b>Batas klaim:</b> dataset tidak memiliki umur, gender, pendapatan, atau psikografis.
    Jadi segmentasi yang sah adalah segmentasi perilaku transaksi berbasis RFM dan geografis sederhana berdasarkan Country.
    """)

    st.subheader("Strategic Meaning of Segments")
    s1, s2, s3 = st.columns(3)
    with s1: section_note("success", "<h4>Champions</h4><b>Pelanggan terbaik.</b><br>Pertahankan dengan loyalty program dan pelayanan prioritas.")
    with s2: section_note("warning", "<h4>High Value At Risk</h4><b>Bernilai tinggi tetapi mulai pasif.</b><br>Prioritas reaktivasi personal.")
    with s3: section_note("info", "<h4>Regular & Lost</h4><b>Perlu strategi hemat biaya.</b><br>Gunakan engagement bertahap dan win-back campaign terkontrol.")

    col1, col2 = st.columns(2)
    if "RFM_Segment" in filtered_customer.columns:
        segment_count = filtered_customer.groupby("RFM_Segment", as_index=False).agg(Customers=("CustomerID", "count"))
        segment_count["RFM_Segment"] = pd.Categorical(segment_count["RFM_Segment"], categories=SEGMENT_ORDER, ordered=True)
        segment_count = segment_count.sort_values("RFM_Segment")
        with col1:
            st.plotly_chart(make_bar(segment_count, "RFM_Segment", "Customers", "Customer per RFM Segment", order=SEGMENT_ORDER), use_container_width=True)

    if {"RFM_Segment", "Monetary"}.issubset(filtered_customer.columns):
        segment_revenue = filtered_customer.groupby("RFM_Segment", as_index=False).agg(Monetary=("Monetary", "sum"))
        segment_revenue["RFM_Segment"] = pd.Categorical(segment_revenue["RFM_Segment"], categories=SEGMENT_ORDER, ordered=True)
        segment_revenue = segment_revenue.sort_values("RFM_Segment")
        with col2:
            st.plotly_chart(make_bar(segment_revenue, "RFM_Segment", "Monetary", "Total Monetary per RFM Segment", order=SEGMENT_ORDER), use_container_width=True)

    c1, c2 = st.columns([0.9, 1.1])
    with c1:
        if "Cluster_Label" in filtered_customer.columns:
            cluster_dist = filtered_customer.groupby("Cluster_Label", as_index=False).agg(Customers=("CustomerID", "count")).sort_values("Customers", ascending=False)
            fig = px.pie(cluster_dist, names="Cluster_Label", values="Customers", hole=0.55, title="Cluster Distribution", template=PLOTLY_TEMPLATE, color_discrete_sequence=[PRIMARY_COLOR, "#0b72c9", "#fca5a5"])
            fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=40), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("RFM Segment vs Cluster")
        if rfm_cluster_matrix is not None:
            matrix_display = rfm_cluster_matrix.copy()
            if "RFM_Segment" in matrix_display.columns:
                matrix_display["RFM_Segment"] = pd.Categorical(matrix_display["RFM_Segment"], categories=SEGMENT_ORDER, ordered=True)
                matrix_display = matrix_display.sort_values("RFM_Segment")
            display_df(rename_for_display(matrix_display), height=420)

    st.subheader("Customer Detail")
    detail_cols = [c for c in ["CustomerID", "RFM_Segment", "Cluster_Label", "Recency", "Frequency", "Monetary", "Business_Priority", "Recommended_Action"] if c in filtered_customer.columns]
    customer_detail = filtered_customer[detail_cols].copy()
    if "Monetary" in customer_detail.columns:
        customer_detail = customer_detail.sort_values("Monetary", ascending=False)
    display_df(rename_for_display(customer_detail.head(30)), height=460)

    if segmentation_workflow is not None:
        st.subheader("Segmentation Workflow")
        display_df(rename_for_display(segmentation_workflow), height=320)

elif page == "4.4 Pemodelan Data Bisnis":
    st.title("4.4 Pemodelan Data Bisnis")
    st.markdown('<div class="subtitle">Fokus bukti: risiko bisnis, value, business score, priority, dan recommended action.</div>', unsafe_allow_html=True)
    st.caption(f"Filter aktif: {filter_status}")

    st.subheader("Indikator dan Bukti")
    display_df(course_indicator_table("Pemodelan Data Bisnis"))

    section_note("warning", """
    <b>Kritik yang harus kamu tulis jujur:</b> ini bukan model prediksi churn.
    Dataset tidak memiliki label churn aktual. Model ini adalah rule-based business decision model
    yang memakai Recency sebagai proxy risiko pasif dan Monetary sebagai proxy customer value.
    """)

    avg_business_score = filtered_customer["Business_Score"].mean() if "Business_Score" in filtered_customer.columns else 0
    total_monetary = filtered_customer["Monetary"].sum() if "Monetary" in filtered_customer.columns else 0
    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Filtered Customers", int_fmt(filtered_customer["CustomerID"].nunique()))
    with c2: kpi_card("Avg Business Score", money_fmt(avg_business_score))
    with c3: kpi_card("Total Monetary", money_fmt(total_monetary))

    st.subheader("Strategic Action by Priority")
    a1, a2, a3 = st.columns(3)
    with a1: section_note("warning", "<h4>Priority 1</h4><b>High Value At Risk</b><br>Reaktivasi personal untuk pelanggan bernilai tinggi yang mulai pasif.")
    with a2: section_note("success", "<h4>Priority 2</h4><b>Champions</b><br>Pertahankan pelanggan terbaik dengan loyalty program.")
    with a3: section_note("info", "<h4>Priority 3</h4><b>Big Spenders & Loyal Customers</b><br>Tingkatkan nilai transaksi melalui cross-selling dan bundling.")

    if "Business_Priority" in filtered_customer.columns:
        prio_summary = (
            filtered_customer.groupby("Business_Priority", as_index=False)
            .agg(Customers=("CustomerID", "count"), Total_Monetary=("Monetary", "sum"), Avg_Business_Score=("Business_Score", "mean"))
            .round(2)
        )
        prio_summary["Business_Priority"] = pd.Categorical(prio_summary["Business_Priority"], categories=PRIORITY_ORDER, ordered=True)
        prio_summary = prio_summary.sort_values("Business_Priority")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(make_bar(prio_summary, "Business_Priority", "Customers", "Customer per Business Priority", order=PRIORITY_ORDER), use_container_width=True)
        with col2:
            st.plotly_chart(make_bar(prio_summary, "Business_Priority", "Total_Monetary", "Monetary per Business Priority", order=PRIORITY_ORDER), use_container_width=True)

    st.subheader("Recommended Action Summary")
    if business_summary is not None:
        bs = business_summary.copy()
        if "Business_Priority" in bs.columns:
            bs["Business_Priority"] = pd.Categorical(bs["Business_Priority"], categories=PRIORITY_ORDER, ordered=True)
            bs = bs.sort_values("Business_Priority")
        display_df(rename_for_display(bs), height=360)

    section_note("warning", """
    <b>Interpretasi penting:</b> Priority 1 bukan berarti revenue terbesar.
    Priority 1 berarti paling urgent karena pelanggan bernilai tinggi mulai pasif.
    Priority 2 adalah Champions, yaitu sumber revenue utama yang harus dipertahankan.
    """)

    if business_modeling_workflow is not None:
        st.subheader("Business Modeling Workflow")
        display_df(rename_for_display(business_modeling_workflow), height=320)

elif page == "4.5 Intuisi dan Wawasan Data":
    st.title("4.5 Intuisi dan Wawasan Data")
    st.markdown('<div class="subtitle">Fokus bukti: insight, interpretasi, keputusan, rekomendasi, dan batasan analisis.</div>', unsafe_allow_html=True)
    st.caption(f"Filter aktif: {filter_status}")

    st.subheader("Indikator dan Bukti")
    display_df(course_indicator_table("Intuisi dan Wawasan Data"))

    section_note("info", """
    <b>Insight utama:</b> jumlah pelanggan saja tidak cukup untuk menentukan strategi.
    Champions adalah revenue driver, High Value At Risk adalah target reaktivasi, sedangkan Lost Customers butuh campaign hemat biaya.
    """)

    if len(filtered_sales) > 0:
        top_country_name = "-"
        top_month = "-"
        top_product = "-"
        if "Country" in filtered_sales.columns and filtered_sales["Country"].notna().any():
            top_country_name = filtered_sales.groupby("Country")["Revenue"].sum().sort_values(ascending=False).index[0]
        if {"MonthYear", "Revenue"}.issubset(filtered_sales.columns):
            top_month = filtered_sales.groupby("MonthYear")["Revenue"].sum().sort_values(ascending=False).index[0]
        if "ProductName" in filtered_sales.columns:
            product_source = remove_non_product(filtered_sales, "ProductName")
            p = product_source.groupby("ProductName")["Revenue"].sum().sort_values(ascending=False)
            if len(p) > 0:
                top_product = p.index[0]

        insight_table = pd.DataFrame({
            "Aspek": ["Total Revenue", "Top Month", "Top Country", "Top Consistent Product", "Main Customer Logic"],
            "Insight": [
                money_fmt(total_revenue),
                top_month,
                top_country_name,
                top_product,
                "Pertahankan Champions, reaktivasi High Value At Risk, hemat biaya untuk Lost Customers"
            ]
        })
        display_df(insight_table)

    if {"MonthYear", "Revenue"}.issubset(filtered_sales.columns):
        monthly = (
            filtered_sales.groupby(["Year", "Month", "MonthYear"], as_index=False)
            .agg(Revenue=("Revenue", "sum"))
            .sort_values(["Year", "Month"])
        )
        fig = px.line(monthly, x="MonthYear", y="Revenue", markers=True, title="Monthly Revenue Trend", template=PLOTLY_TEMPLATE, color_discrete_sequence=[PRIMARY_COLOR])
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=40), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    section_note("warning", """
    <b>Batasan:</b> Revenue bukan profit karena dataset tidak memiliki biaya/margin.
    Data Desember 2011 tidak penuh satu bulan. Tidak ada data demografis atau label churn aktual.
    """)

    limitations_path = Path(data_dir) / "limitations_project1.txt"
    if limitations_path.exists():
        st.subheader("Limitations")
        st.text(limitations_path.read_text(encoding="utf-8"))

elif page == "Lampiran Data Model":
    st.title("Lampiran Data Model")
    st.markdown('<div class="subtitle">Halaman pendukung untuk screenshot star schema, validation, dan course mapping.</div>', unsafe_allow_html=True)

    st.subheader("Star Schema")
    star_schema = pd.DataFrame({
        "Table": ["fact_sales", "dim_customer", "dim_product", "dim_country", "dim_date"],
        "Role": ["Fact table", "Customer dimension", "Product dimension", "Country dimension", "Date dimension"],
        "Key": ["CustomerID, ProductKey, CountryKey, DateKey", "CustomerID", "ProductKey", "CountryKey", "DateKey"],
        "Rows": [len(fact_sales), len(dim_customer), len(dim_product) if dim_product is not None else 0, len(dim_country) if dim_country is not None else 0, len(dim_date) if dim_date is not None else 0],
    })
    display_df(star_schema)

    st.graphviz_chart("""
    digraph {
        graph [rankdir=LR, bgcolor="transparent"]
        node [shape=box, style="rounded,filled", fillcolor="#1f2937", fontcolor="white", color="#ff4b4b"]
        edge [fontcolor="white", color="#9ca3af"]
        dim_customer [label="dim_customer\\nCustomerID"]
        dim_product [label="dim_product\\nProductKey"]
        dim_country [label="dim_country\\nCountryKey"]
        dim_date [label="dim_date\\nDateKey"]
        fact_sales [label="fact_sales\\nCustomerID, ProductKey, CountryKey, DateKey"]
        dim_customer -> fact_sales [label="1 : many"]
        dim_product -> fact_sales [label="1 : many"]
        dim_country -> fact_sales [label="1 : many"]
        dim_date -> fact_sales [label="1 : many"]
    }
    """)

    st.subheader("Relationship")
    relationship = pd.DataFrame({
        "Relationship": [
            "dim_customer[CustomerID] → fact_sales[CustomerID]",
            "dim_product[ProductKey] → fact_sales[ProductKey]",
            "dim_country[CountryKey] → fact_sales[CountryKey]",
            "dim_date[DateKey] → fact_sales[DateKey]",
        ],
        "Cardinality": ["One to Many", "One to Many", "One to Many", "One to Many"],
        "Purpose": [
            "Menghubungkan profil dan segmentasi pelanggan dengan transaksi",
            "Menghubungkan atribut produk dengan transaksi",
            "Menghubungkan negara pelanggan dengan transaksi",
            "Menghubungkan kalender waktu dengan transaksi",
        ],
    })
    display_df(relationship)

    st.subheader("Data Validation")
    validation = pd.DataFrame({
        "Validation Check": ["fact_sales rows", "dim_customer rows", "dim_product rows", "dim_country rows", "dim_date rows", "Missing CustomerID", "Missing ProductKey", "Missing CountryKey", "Missing DateKey"],
        "Result": [
            len(fact_sales), len(dim_customer), len(dim_product) if dim_product is not None else 0, len(dim_country) if dim_country is not None else 0, len(dim_date) if dim_date is not None else 0,
            fact_sales["CustomerID"].isna().sum() if "CustomerID" in fact_sales.columns else "-",
            fact_sales["ProductKey"].isna().sum() if "ProductKey" in fact_sales.columns else "-",
            fact_sales["CountryKey"].isna().sum() if "CountryKey" in fact_sales.columns else "-",
            fact_sales["DateKey"].isna().sum() if "DateKey" in fact_sales.columns else "-",
        ],
    })
    display_df(validation)

    st.subheader("Course Mapping")
    if course_mapping is not None:
        display_df(rename_for_display(course_mapping), height=420)

    section_note("info", """
    <b>Catatan screenshot:</b> untuk laporan, crop area dashboard saja. Jangan masukkan address bar, sidebar, atau Manage App.
    """)