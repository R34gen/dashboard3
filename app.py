# ============================================================
# PROJECT 1 — CUSTOMER INTELLIGENCE DASHBOARD
# Streamlit Final v2
# ============================================================
# Fokus:
# - KPI penjualan
# - Segmentasi pelanggan berbasis RFM
# - Clustering sebagai pendukung
# - Business priority dan recommended action
# - Data model, workflow, method decision, limitations
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Project 1 | Customer Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLE
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --bg: #0e1117;
        --panel: #161b22;
        --panel2: #1f2937;
        --text: #f9fafb;
        --muted: #9ca3af;
        --accent: #ff4b4b;
        --blue: #7cc4ff;
        --green: #22c55e;
        --orange: #f97316;
        --border: rgba(255,255,255,0.10);
    }

    .main .block-container {
        padding-top: 2.0rem;
        padding-bottom: 4rem;
        max-width: 1400px;
    }

    h1, h2, h3 {
        letter-spacing: -0.04em;
    }

    .subtitle {
        color: var(--muted);
        font-size: 1.05rem;
        margin-top: -0.6rem;
        margin-bottom: 1.6rem;
    }

    .kpi-card {
        background: linear-gradient(135deg, #ffffff 0%, #f5f7fb 100%);
        color: #111827;
        border-radius: 18px;
        padding: 22px 24px;
        border: 1px solid rgba(255,255,255,0.08);
        min-height: 125px;
        box-shadow: 0 16px 40px rgba(0,0,0,0.18);
    }

    .kpi-label {
        color: #6b7280;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .kpi-value {
        color: #111827;
        font-size: 2.0rem;
        font-weight: 800;
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

    .small-muted {
        color: #9ca3af;
        font-size: 0.92rem;
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
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================
SEGMENT_ORDER = [
    "Champions",
    "Loyal Customers",
    "Big Spenders",
    "High Value At Risk",
    "At Risk",
    "Regular Customers",
    "Lost Customers",
]

PRIORITY_ORDER = [
    "Priority 1",
    "Priority 2",
    "Priority 3",
    "Priority 4",
    "Priority 5",
    "Priority 6",
]

NON_PRODUCT = [
    "POSTAGE",
    "DOTCOM POSTAGE",
    "CARRIAGE",
    "MANUAL",
    "BANK CHARGES",
    "AMAZON FEE",
    "SAMPLES",
    "ADJUST",
    "ADJUSTMENT",
]

PLOTLY_TEMPLATE = "plotly_dark"
PRIMARY_COLOR = "#7cc4ff"
ACCENT_COLOR = "#ff4b4b"
GREEN_COLOR = "#22c55e"
ORANGE_COLOR = "#f97316"


# ============================================================
# HELPERS
# ============================================================
def money_fmt(x):
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return "-"


def int_fmt(x):
    try:
        return f"{int(x):,}"
    except Exception:
        return "-"


def safe_col(df, col, default=None):
    if df is not None and col in df.columns:
        return df[col]
    return default


@st.cache_data(show_spinner=False)
def load_csv(filename: str, data_dir: str) -> pd.DataFrame | None:
    path = Path(data_dir) / filename
    if not path.exists():
        return None
    return pd.read_csv(path)


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return df
    out = df.copy()
    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].astype(str).str.strip()
    return out


def display_df(df: pd.DataFrame, *, height=None):
    """
    Wrapper aman untuk st.dataframe.
    Streamlit versi baru tidak menerima height=None, jadi parameter height
    hanya dikirim jika benar-benar diberi nilai.
    """
    if height is None:
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True, height=height)


def kpi_card(label: str, value: str):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_note(kind: str, html: str):
    css = {
        "info": "insight-box",
        "success": "success-box",
        "warning": "warning-box",
    }.get(kind, "insight-box")

    st.markdown(f'<div class="{css}">{html}</div>', unsafe_allow_html=True)


def rename_for_display(df: pd.DataFrame) -> pd.DataFrame:
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
    }
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})


def make_bar(df, x, y, title, order=None, text=None):
    fig = px.bar(
        df,
        x=x,
        y=y,
        text=text,
        title=title,
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[PRIMARY_COLOR],
    )
    if order:
        fig.update_xaxes(categoryorder="array", categoryarray=order)
    fig.update_layout(
        title_font_size=18,
        margin=dict(l=20, r=20, t=60, b=40),
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_hbar(df, x, y, title):
    fig = px.bar(
        df,
        x=x,
        y=y,
        orientation="h",
        title=title,
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[PRIMARY_COLOR],
    )
    fig.update_layout(
        title_font_size=18,
        margin=dict(l=20, r=20, t=60, b=40),
        height=460,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed"),
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

    if dim_customer is not None and {"CustomerID", "RFM_Segment", "Business_Priority"}.issubset(dim_customer.columns):
        cust_cols = [
            c for c in [
                "CustomerID", "RFM_Segment", "Cluster_Label", "Business_Priority",
                "Recommended_Action", "Business_Score"
            ] if c in dim_customer.columns
        ]
        sales = sales.merge(dim_customer[cust_cols], on="CustomerID", how="left")

    return sales


def remove_non_product(df, product_col):
    out = df.copy()
    if product_col in out.columns:
        mask = ~out[product_col].astype(str).str.upper().str.strip().isin(NON_PRODUCT)
        out = out[mask]
    return out


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Project 1 Dashboard")
st.sidebar.caption("Customer Intelligence Online Retail")

data_dir = st.sidebar.text_input(
    "Data folder",
    value="project1_output",
    help="Folder berisi hasil export final dari Colab.",
)

page = st.sidebar.radio(
    "Pilih halaman",
    [
        "Overview",
        "Sales Performance",
        "Customer Segmentation",
        "Business Recommendation",
        "Data Model & Methodology",
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

kpi_file = load_csv("sales_kpi.csv", data_dir)
monthly_sales_file = load_csv("monthly_sales.csv", data_dir)
top_country_file = load_csv("top_country.csv", data_dir)
top_product_clean_file = load_csv("top_product_clean.csv", data_dir)
top_customer_file = load_csv("top_customer.csv", data_dir)

business_summary = clean_text_columns(load_csv("business_summary.csv", data_dir))
priority_summary = clean_text_columns(load_csv("priority_summary.csv", data_dir))
segment_summary = clean_text_columns(load_csv("rfm_segment_summary.csv", data_dir))
cluster_profile = clean_text_columns(load_csv("cluster_profile.csv", data_dir))
rfm_cluster_matrix = clean_text_columns(load_csv("rfm_cluster_matrix.csv", data_dir))

bi_workflow = clean_text_columns(load_csv("bi_workflow.csv", data_dir))
segmentation_workflow = clean_text_columns(load_csv("segmentation_workflow.csv", data_dir))
business_modeling_workflow = clean_text_columns(load_csv("business_modeling_workflow.csv", data_dir))
method_decision = clean_text_columns(load_csv("method_decision.csv", data_dir))
course_mapping = clean_text_columns(load_csv("course_mapping_project1.csv", data_dir))

if fact_sales is None or dim_customer is None:
    st.error(
        "File utama belum ditemukan. Pastikan folder project1_output berisi minimal fact_sales.csv dan dim_customer.csv."
    )
    st.stop()

# Type normalization
for col in ["Revenue", "Quantity", "UnitPrice"]:
    if col in fact_sales.columns:
        fact_sales[col] = pd.to_numeric(fact_sales[col], errors="coerce")

for col in ["Monetary", "Recency", "Frequency", "Business_Score"]:
    if col in dim_customer.columns:
        dim_customer[col] = pd.to_numeric(dim_customer[col], errors="coerce")

if dim_date is not None:
    for col in ["Year", "Month", "DateKey"]:
        if col in dim_date.columns:
            dim_date[col] = pd.to_numeric(dim_date[col], errors="coerce")
    if "Date" in dim_date.columns:
        dim_date["Date"] = pd.to_datetime(dim_date["Date"], errors="coerce")

sales_model = prepare_sales_model(fact_sales, dim_customer, dim_product, dim_country, dim_date)

# Sidebar filter values
years = sorted([int(y) for y in sales_model["Year"].dropna().unique()]) if "Year" in sales_model.columns else []
countries = sorted(sales_model["Country"].dropna().unique()) if "Country" in sales_model.columns else []
segments = [s for s in SEGMENT_ORDER if s in dim_customer.get("RFM_Segment", pd.Series(dtype=str)).dropna().unique()]
priorities = [p for p in PRIORITY_ORDER if p in dim_customer.get("Business_Priority", pd.Series(dtype=str)).dropna().unique()]

selected_year = st.sidebar.multiselect("Year", years, default=[])
selected_country = st.sidebar.multiselect("Country", countries, default=[])
selected_segment = st.sidebar.multiselect("RFM Segment", segments, default=[])
selected_priority = st.sidebar.multiselect("Business Priority", priorities, default=[])

filtered_customer = dim_customer.copy()

if selected_segment and "RFM_Segment" in filtered_customer.columns:
    filtered_customer = filtered_customer[filtered_customer["RFM_Segment"].isin(selected_segment)]

if selected_priority and "Business_Priority" in filtered_customer.columns:
    filtered_customer = filtered_customer[filtered_customer["Business_Priority"].isin(selected_priority)]

allowed_customers = set(filtered_customer["CustomerID"].astype(str)) if "CustomerID" in filtered_customer.columns else set()

filtered_sales = sales_model.copy()
if "CustomerID" in filtered_sales.columns and allowed_customers:
    filtered_sales = filtered_sales[filtered_sales["CustomerID"].astype(str).isin(allowed_customers)]

if selected_year and "Year" in filtered_sales.columns:
    filtered_sales = filtered_sales[filtered_sales["Year"].isin(selected_year)]

if selected_country and "Country" in filtered_sales.columns:
    filtered_sales = filtered_sales[filtered_sales["Country"].isin(selected_country)]


# Sidebar download
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


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================
if page == "Overview":
    st.title("📊 Customer Intelligence Dashboard")
    st.markdown(
        '<div class="subtitle">Project 1 — Online Retail Customer Intelligence untuk MBKM Studi Independen</div>',
        unsafe_allow_html=True,
    )

    total_revenue = filtered_sales["Revenue"].sum() if "Revenue" in filtered_sales.columns else 0
    total_transaction = filtered_sales["InvoiceNo"].nunique() if "InvoiceNo" in filtered_sales.columns else 0
    total_customer = filtered_sales["CustomerID"].nunique() if "CustomerID" in filtered_sales.columns else 0
    total_product = filtered_sales["ProductKey"].nunique() if "ProductKey" in filtered_sales.columns else 0
    aov = total_revenue / total_transaction if total_transaction else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Total Revenue", money_fmt(total_revenue))
    with c2:
        kpi_card("Transactions", int_fmt(total_transaction))
    with c3:
        kpi_card("Customers", int_fmt(total_customer))
    with c4:
        kpi_card("Products", int_fmt(total_product))
    with c5:
        kpi_card("Avg Order Value", money_fmt(aov))

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    left, right = st.columns([1.15, 1])

    with left:
        st.subheader("Executive Summary")
        section_note(
            "info",
            """
            <b>1. Revenue didominasi oleh Champions.</b><br>
            Segmen Champions menjadi kontributor revenue terbesar sehingga strategi utama adalah mempertahankan pelanggan ini melalui loyalty program.
            <br><br>
            <b>2. High Value At Risk menjadi prioritas reaktivasi.</b><br>
            Segmen ini memiliki nilai pembelian tinggi tetapi mulai pasif, sehingga perlu pendekatan personal.
            <br><br>
            <b>3. Lost Customers tidak boleh menjadi target promosi mahal.</b><br>
            Jumlahnya besar, tetapi kontribusi revenue relatif lebih rendah sehingga campaign perlu dibuat hemat biaya.
            """,
        )

    with right:
        st.subheader("Output Utama")
        st.markdown(
            """
            - Sales Performance
            - Customer Segmentation
            - Business Recommendation
            - Data Model & Methodology
            """
        )
        section_note(
            "success",
            """
            Dashboard ini digunakan sebagai reporting tool interaktif berbasis Python.
            Fokusnya bukan hanya visualisasi, tetapi mendukung pengambilan keputusan bisnis berbasis data.
            """,
        )


# ============================================================
# PAGE 2 — SALES PERFORMANCE
# ============================================================
elif page == "Sales Performance":
    st.title("📈 Sales Performance")
    st.markdown(
        '<div class="subtitle">KPI, tren revenue, negara, produk, dan customer dengan kontribusi tertinggi.</div>',
        unsafe_allow_html=True,
    )

    total_revenue = filtered_sales["Revenue"].sum()
    total_transaction = filtered_sales["InvoiceNo"].nunique()
    total_customer = filtered_sales["CustomerID"].nunique()
    aov = total_revenue / total_transaction if total_transaction else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Revenue", money_fmt(total_revenue))
    with c2:
        kpi_card("Transactions", int_fmt(total_transaction))
    with c3:
        kpi_card("Customers", int_fmt(total_customer))
    with c4:
        kpi_card("AOV", money_fmt(aov))

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if {"MonthYear", "Revenue"}.issubset(filtered_sales.columns):
        monthly = (
            filtered_sales.groupby(["Year", "Month", "MonthYear"], as_index=False)
            .agg(Revenue=("Revenue", "sum"))
            .sort_values(["Year", "Month"])
        )

        fig = px.line(
            monthly,
            x="MonthYear",
            y="Revenue",
            markers=True,
            title="Monthly Revenue Trend",
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=[PRIMARY_COLOR],
        )
        fig.update_layout(
            height=420,
            margin=dict(l=20, r=20, t=60, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        section_note(
            "warning",
            """
            <b>Catatan interpretasi:</b> Data Desember 2011 hanya tersedia sampai tanggal 9,
            sehingga penurunan revenue pada Desember tidak boleh dibaca sebagai performa satu bulan penuh.
            """,
        )

    col1, col2 = st.columns(2)

    with col1:
        if "Country" in filtered_sales.columns:
            country_chart = (
                filtered_sales.groupby("Country", as_index=False)
                .agg(Revenue=("Revenue", "sum"))
                .sort_values("Revenue", ascending=False)
                .head(10)
            )
            fig = make_bar(country_chart, "Country", "Revenue", "Top 10 Country by Revenue")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Always recompute from filtered_sales, and force remove non-product items.
        if "ProductName" in filtered_sales.columns:
            product_source = remove_non_product(filtered_sales, "ProductName")
            product_chart = (
                product_source.groupby("ProductName", as_index=False)
                .agg(
                    Revenue=("Revenue", "sum"),
                    Transaction=("InvoiceNo", "nunique"),
                )
                .query("Transaction >= 50")
                .sort_values("Revenue", ascending=False)
                .head(10)
            )
            fig = make_hbar(
                product_chart,
                "Revenue",
                "ProductName",
                "Top 10 Consistent Product by Revenue",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Customer by Revenue")
    if {"CustomerID", "Revenue", "InvoiceNo"}.issubset(filtered_sales.columns):
        top_customer_dashboard = (
            filtered_sales.groupby("CustomerID", as_index=False)
            .agg(
                Revenue=("Revenue", "sum"),
                Transaction=("InvoiceNo", "nunique"),
            )
            .sort_values("Revenue", ascending=False)
            .head(10)
        )
        display_df(rename_for_display(top_customer_dashboard))


# ============================================================
# PAGE 3 — CUSTOMER SEGMENTATION
# ============================================================
elif page == "Customer Segmentation":
    st.title("👥 Customer Segmentation")
    st.markdown(
        '<div class="subtitle">RFM Segment sebagai dasar strategi bisnis; clustering digunakan sebagai validasi pola data.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    if "RFM_Segment" in filtered_customer.columns:
        segment_count = (
            filtered_customer.groupby("RFM_Segment", as_index=False)
            .agg(Customers=("CustomerID", "count"))
        )
        segment_count["RFM_Segment"] = pd.Categorical(segment_count["RFM_Segment"], categories=SEGMENT_ORDER, ordered=True)
        segment_count = segment_count.sort_values("RFM_Segment")

        with col1:
            fig = make_bar(
                segment_count,
                "RFM_Segment",
                "Customers",
                "Customer per RFM Segment",
                order=SEGMENT_ORDER,
            )
            st.plotly_chart(fig, use_container_width=True)

    if {"RFM_Segment", "Monetary"}.issubset(filtered_customer.columns):
        segment_revenue = (
            filtered_customer.groupby("RFM_Segment", as_index=False)
            .agg(Monetary=("Monetary", "sum"))
        )
        segment_revenue["RFM_Segment"] = pd.Categorical(segment_revenue["RFM_Segment"], categories=SEGMENT_ORDER, ordered=True)
        segment_revenue = segment_revenue.sort_values("RFM_Segment")

        with col2:
            fig = make_bar(
                segment_revenue,
                "RFM_Segment",
                "Monetary",
                "Total Monetary per RFM Segment",
                order=SEGMENT_ORDER,
            )
            st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns([0.9, 1.1])

    with c1:
        if "Cluster_Label" in filtered_customer.columns:
            cluster_dist = (
                filtered_customer.groupby("Cluster_Label", as_index=False)
                .agg(Customers=("CustomerID", "count"))
                .sort_values("Customers", ascending=False)
            )
            fig = px.pie(
                cluster_dist,
                names="Cluster_Label",
                values="Customers",
                hole=0.55,
                title="Cluster Distribution",
                template=PLOTLY_TEMPLATE,
                color_discrete_sequence=[PRIMARY_COLOR, "#0b72c9", "#fca5a5"],
            )
            fig.update_layout(
                height=420,
                margin=dict(l=20, r=20, t=60, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("RFM Segment vs Cluster")
        if rfm_cluster_matrix is not None:
            matrix_display = rfm_cluster_matrix.copy()
            if "RFM_Segment" in matrix_display.columns:
                matrix_display["RFM_Segment"] = pd.Categorical(matrix_display["RFM_Segment"], categories=SEGMENT_ORDER, ordered=True)
                matrix_display = matrix_display.sort_values("RFM_Segment")
            display_df(rename_for_display(matrix_display), height=420)
        else:
            st.caption("rfm_cluster_matrix.csv tidak ditemukan.")

    st.subheader("Customer Detail")
    detail_cols = [
        "CustomerID", "RFM_Segment", "Cluster_Label",
        "Recency", "Frequency", "Monetary",
        "Business_Priority", "Recommended_Action",
    ]
    detail_cols = [c for c in detail_cols if c in filtered_customer.columns]

    customer_detail = (
        filtered_customer[detail_cols]
        .sort_values("Monetary", ascending=False)
        .head(30)
        if "Monetary" in filtered_customer.columns
        else filtered_customer[detail_cols].head(30)
    )
    display_df(rename_for_display(customer_detail), height=460)

    section_note(
        "warning",
        """
        <b>Catatan:</b> RFM Segment digunakan sebagai dasar rekomendasi bisnis.
        Cluster hanya menjadi pendukung karena clustering dapat terpengaruh nilai Monetary ekstrem.
        """,
    )


# ============================================================
# PAGE 4 — BUSINESS RECOMMENDATION
# ============================================================
elif page == "Business Recommendation":
    st.title("🎯 Business Recommendation")
    st.markdown(
        '<div class="subtitle">Prioritas bisnis berdasarkan RFM Segment, Risk Score, Value Score, dan Business Score.</div>',
        unsafe_allow_html=True,
    )

    total_customer = filtered_customer["CustomerID"].nunique() if "CustomerID" in filtered_customer.columns else 0
    avg_business_score = filtered_customer["Business_Score"].mean() if "Business_Score" in filtered_customer.columns else 0
    total_monetary = filtered_customer["Monetary"].sum() if "Monetary" in filtered_customer.columns else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Filtered Customers", int_fmt(total_customer))
    with c2:
        kpi_card("Avg Business Score", money_fmt(avg_business_score))
    with c3:
        kpi_card("Total Monetary", money_fmt(total_monetary))

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.subheader("Strategic Action by Priority")
    a1, a2, a3 = st.columns(3)

    with a1:
        section_note(
            "warning",
            """
            <h4>Priority 1</h4>
            <b>High Value At Risk</b><br>
            Reaktivasi personal untuk pelanggan bernilai tinggi yang mulai pasif.
            """,
        )
    with a2:
        section_note(
            "success",
            """
            <h4>Priority 2</h4>
            <b>Champions</b><br>
            Pertahankan pelanggan terbaik dengan loyalty program.
            """,
        )
    with a3:
        section_note(
            "info",
            """
            <h4>Priority 3</h4>
            <b>Big Spenders & Loyal Customers</b><br>
            Tingkatkan nilai transaksi melalui cross-selling dan bundling.
            """,
        )

    if "Business_Priority" in filtered_customer.columns:
        prio_summary = (
            filtered_customer.groupby("Business_Priority", as_index=False)
            .agg(
                Customers=("CustomerID", "count"),
                Total_Monetary=("Monetary", "sum"),
                Avg_Business_Score=("Business_Score", "mean"),
            )
            .round(2)
        )
        prio_summary["Business_Priority"] = pd.Categorical(prio_summary["Business_Priority"], categories=PRIORITY_ORDER, ordered=True)
        prio_summary = prio_summary.sort_values("Business_Priority")

        col1, col2 = st.columns(2)

        with col1:
            fig = make_bar(
                prio_summary,
                "Business_Priority",
                "Customers",
                "Customer per Business Priority",
                order=PRIORITY_ORDER,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = make_bar(
                prio_summary,
                "Business_Priority",
                "Total_Monetary",
                "Monetary per Business Priority",
                order=PRIORITY_ORDER,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recommended Action Summary")
    if business_summary is not None:
        bs = business_summary.copy()
        if "Business_Priority" in bs.columns:
            bs["Business_Priority"] = pd.Categorical(bs["Business_Priority"], categories=PRIORITY_ORDER, ordered=True)
            bs = bs.sort_values("Business_Priority")
        display_df(rename_for_display(bs), height=360)
    else:
        display_df(rename_for_display(prio_summary))

    section_note(
        "warning",
        """
        <b>Interpretasi penting:</b> Priority 1 bukan berarti revenue terbesar.
        Priority 1 berarti paling urgent karena pelanggan bernilai tinggi mulai pasif.
        Priority 2 adalah Champions, yaitu sumber revenue utama yang harus dipertahankan.
        """,
    )


# ============================================================
# PAGE 5 — DATA MODEL & METHODOLOGY
# ============================================================
elif page == "Data Model & Methodology":
    st.title("🧩 Data Model & Methodology")
    st.markdown(
        '<div class="subtitle">Bukti model data, relasi, workflow, metode, validasi, dan batasan analisis.</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Star Schema")
    star_schema = pd.DataFrame({
        "Table": ["fact_sales", "dim_customer", "dim_product", "dim_country", "dim_date"],
        "Role": ["Fact table", "Customer dimension", "Product dimension", "Country dimension", "Date dimension"],
        "Key": [
            "CustomerID, ProductKey, CountryKey, DateKey",
            "CustomerID",
            "ProductKey",
            "CountryKey",
            "DateKey",
        ],
        "Rows": [
            len(fact_sales),
            len(dim_customer) if dim_customer is not None else 0,
            len(dim_product) if dim_product is not None else 0,
            len(dim_country) if dim_country is not None else 0,
            len(dim_date) if dim_date is not None else 0,
        ],
    })
    display_df(star_schema)

    st.subheader("Star Schema Diagram")
    st.graphviz_chart(
        """
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
        """
    )

    st.subheader("Relationship")
    relationship = pd.DataFrame({
        "Relationship": [
            "fact_sales[CustomerID] → dim_customer[CustomerID]",
            "fact_sales[ProductKey] → dim_product[ProductKey]",
            "fact_sales[CountryKey] → dim_country[CountryKey]",
            "fact_sales[DateKey] → dim_date[DateKey]",
        ],
        "Cardinality": ["Many to One", "Many to One", "Many to One", "Many to One"],
        "Purpose": [
            "Menghubungkan transaksi dengan profil dan segmentasi pelanggan",
            "Menghubungkan transaksi dengan atribut produk",
            "Menghubungkan transaksi dengan negara pelanggan",
            "Menghubungkan transaksi dengan kalender analisis waktu",
        ],
    })
    display_df(relationship)

    st.subheader("Data Validation")
    validation = pd.DataFrame({
        "Validation Check": [
            "fact_sales rows",
            "dim_customer rows",
            "dim_product rows",
            "dim_country rows",
            "dim_date rows",
            "Missing CustomerID",
            "Missing ProductKey",
            "Missing CountryKey",
            "Missing DateKey",
        ],
        "Result": [
            len(fact_sales),
            len(dim_customer) if dim_customer is not None else 0,
            len(dim_product) if dim_product is not None else 0,
            len(dim_country) if dim_country is not None else 0,
            len(dim_date) if dim_date is not None else 0,
            fact_sales["CustomerID"].isna().sum() if "CustomerID" in fact_sales.columns else "-",
            fact_sales["ProductKey"].isna().sum() if "ProductKey" in fact_sales.columns else "-",
            fact_sales["CountryKey"].isna().sum() if "CountryKey" in fact_sales.columns else "-",
            fact_sales["DateKey"].isna().sum() if "DateKey" in fact_sales.columns else "-",
        ],
    })
    display_df(validation)

    tabs = st.tabs([
        "BI Workflow",
        "Segmentation",
        "Business Modeling",
        "Method Decision",
        "Course Mapping",
        "Limitations",
    ])

    with tabs[0]:
        if bi_workflow is not None:
            workflow = bi_workflow.replace("Power BI", "Streamlit", regex=True)
            display_df(rename_for_display(workflow), height=360)
        else:
            st.caption("bi_workflow.csv tidak ditemukan.")

    with tabs[1]:
        if segmentation_workflow is not None:
            display_df(rename_for_display(segmentation_workflow), height=360)
        else:
            st.caption("segmentation_workflow.csv tidak ditemukan.")

    with tabs[2]:
        if business_modeling_workflow is not None:
            display_df(rename_for_display(business_modeling_workflow), height=360)
        else:
            st.caption("business_modeling_workflow.csv tidak ditemukan.")

    with tabs[3]:
        if method_decision is not None:
            display_df(rename_for_display(method_decision), height=420)
        else:
            st.caption("method_decision.csv tidak ditemukan.")

    with tabs[4]:
        if course_mapping is not None:
            display_df(rename_for_display(course_mapping), height=420)
        else:
            st.caption("course_mapping_project1.csv tidak ditemukan.")

    with tabs[5]:
        limitations_path = Path(data_dir) / "limitations_project1.txt"
        if limitations_path.exists():
            st.text(limitations_path.read_text(encoding="utf-8"))
        else:
            section_note(
                "warning",
                """
                Dataset tidak memiliki data demografis dan label churn aktual.
                Segmentasi dilakukan berdasarkan perilaku transaksi berbasis RFM.
                """
            )