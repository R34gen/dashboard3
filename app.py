# ============================================================
# STREAMLIT DASHBOARD — PROJECT 1 CUSTOMER INTELLIGENCE
# Online Retail Customer Intelligence Dashboard
# ============================================================
# Cara run:
# 1) Letakkan folder project1_output di folder yang sama dengan app.py
# 2) pip install -r requirements.txt
# 3) streamlit run app.py
# ============================================================

from pathlib import Path
import re
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Project 1 | Customer Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Styling sederhana, rapi, tidak berlebihan
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 16px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .metric-label {
        font-size: 0.88rem;
        color: #666;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #111;
    }
    .note-box {
        background-color: #f7f7f8;
        border-left: 4px solid #666;
        padding: 12px 14px;
        border-radius: 8px;
        font-size: 0.95rem;
    }
    .danger-box {
        background-color: #fff7ed;
        border-left: 4px solid #f97316;
        padding: 12px 14px;
        border-radius: 8px;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------
def rupiah_like(x):
    try:
        return f"{x:,.2f}"
    except Exception:
        return "-"


def int_like(x):
    try:
        return f"{int(x):,}"
    except Exception:
        return "-"


def priority_number(value):
    m = re.search(r"(\d+)", str(value))
    return int(m.group(1)) if m else 99


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_read_csv(path, **kwargs):
    if not path.exists():
        return None
    return pd.read_csv(path, **kwargs)


@st.cache_data(show_spinner=False)
def load_data(data_dir: str):
    base = Path(data_dir)

    # Wajib untuk dashboard utama
    fact_sales = safe_read_csv(base / "fact_sales.csv", dtype={"CustomerID": str, "InvoiceNo": str})
    dim_customer = safe_read_csv(base / "dim_customer.csv", dtype={"CustomerID": str})
    dim_product = safe_read_csv(base / "dim_product.csv")
    dim_country = safe_read_csv(base / "dim_country.csv")
    dim_date = safe_read_csv(base / "dim_date.csv")

    # Output pendukung
    sales_kpi = safe_read_csv(base / "sales_kpi.csv")
    monthly_sales = safe_read_csv(base / "monthly_sales.csv")
    top_country = safe_read_csv(base / "top_country.csv")
    top_product_clean = safe_read_csv(base / "top_product_clean.csv")
    top_customer = safe_read_csv(base / "top_customer.csv", dtype={"CustomerID": str})
    segment_summary = safe_read_csv(base / "rfm_segment_summary.csv")
    cluster_profile = safe_read_csv(base / "cluster_profile.csv")
    business_summary = safe_read_csv(base / "business_summary.csv")
    priority_summary = safe_read_csv(base / "priority_summary.csv")
    method_decision = safe_read_csv(base / "method_decision.csv")
    bi_workflow = safe_read_csv(base / "bi_workflow.csv")
    segmentation_workflow = safe_read_csv(base / "segmentation_workflow.csv")
    business_modeling_workflow = safe_read_csv(base / "business_modeling_workflow.csv")
    course_mapping = safe_read_csv(base / "course_mapping_project1.csv")
    silhouette_score = safe_read_csv(base / "silhouette_score.csv")

    limitations = ""
    insight = ""
    if (base / "limitations_project1.txt").exists():
        limitations = (base / "limitations_project1.txt").read_text(encoding="utf-8")
    if (base / "insight_project1.txt").exists():
        insight = (base / "insight_project1.txt").read_text(encoding="utf-8")

    required = {
        "fact_sales": fact_sales,
        "dim_customer": dim_customer,
        "dim_product": dim_product,
        "dim_country": dim_country,
        "dim_date": dim_date,
    }

    return {
        "base": base,
        "required": required,
        "fact_sales": fact_sales,
        "dim_customer": dim_customer,
        "dim_product": dim_product,
        "dim_country": dim_country,
        "dim_date": dim_date,
        "sales_kpi": sales_kpi,
        "monthly_sales": monthly_sales,
        "top_country": top_country,
        "top_product_clean": top_product_clean,
        "top_customer": top_customer,
        "segment_summary": segment_summary,
        "cluster_profile": cluster_profile,
        "business_summary": business_summary,
        "priority_summary": priority_summary,
        "method_decision": method_decision,
        "bi_workflow": bi_workflow,
        "segmentation_workflow": segmentation_workflow,
        "business_modeling_workflow": business_modeling_workflow,
        "course_mapping": course_mapping,
        "silhouette_score": silhouette_score,
        "limitations": limitations,
        "insight": insight,
    }


def validate_required(data):
    missing = [name for name, df in data["required"].items() if df is None]
    if missing:
        st.error(
            "File wajib belum ditemukan: " + ", ".join(missing) +
            ". Pastikan folder project1_output berada satu folder dengan app.py atau ubah path di sidebar."
        )
        st.stop()


def build_sales_model(fact_sales, dim_customer, dim_product, dim_country, dim_date):
    sales = fact_sales.copy()
    customer = dim_customer.copy()
    product = dim_product.copy()
    country = dim_country.copy()
    date = dim_date.copy()

    # Tipe data kunci
    sales["CustomerID"] = sales["CustomerID"].astype(str)
    customer["CustomerID"] = customer["CustomerID"].astype(str)

    if "Date" in date.columns:
        date["Date"] = pd.to_datetime(date["Date"], errors="coerce")

    sales = sales.merge(product[["ProductKey", "ProductName"]], on="ProductKey", how="left")
    sales = sales.merge(country[["CountryKey", "Country"]], on="CountryKey", how="left")
    date_cols = [c for c in ["DateKey", "Date", "Year", "Month", "MonthName", "Quarter", "MonthYear"] if c in date.columns]
    sales = sales.merge(date[date_cols], on="DateKey", how="left")

    customer_cols = [
        "CustomerID", "RFM_Segment", "Cluster_Label", "Business_Priority",
        "Recommended_Action", "Risk_Score", "Value_Score", "Business_Score"
    ]
    customer_cols = [c for c in customer_cols if c in customer.columns]
    sales = sales.merge(customer[customer_cols], on="CustomerID", how="left")

    return sales


# ------------------------------------------------------------
# Sidebar — Data path dan navigasi
# ------------------------------------------------------------
st.sidebar.title("Project 1 Dashboard")
st.sidebar.caption("Customer Intelligence Online Retail")

data_dir = st.sidebar.text_input("Folder output Colab", value="project1_output")
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

data = load_data(data_dir)
validate_required(data)

fact_sales = data["fact_sales"]
dim_customer = data["dim_customer"]
dim_product = data["dim_product"]
dim_country = data["dim_country"]
dim_date = data["dim_date"]

sales_model = build_sales_model(fact_sales, dim_customer, dim_product, dim_country, dim_date)
customer_model = dim_customer.copy()
customer_model["CustomerID"] = customer_model["CustomerID"].astype(str)

# ------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Filter")

# Year filter
if "Year" in sales_model.columns:
    years = sorted([int(y) for y in sales_model["Year"].dropna().unique()])
    selected_years = st.sidebar.multiselect("Year", years, default=years)
else:
    selected_years = []

# Country filter
if "Country" in sales_model.columns:
    countries = sorted(sales_model["Country"].dropna().astype(str).unique())
    default_countries = countries
    selected_countries = st.sidebar.multiselect("Country", countries, default=default_countries)
else:
    selected_countries = []

# Segment filter
if "RFM_Segment" in customer_model.columns:
    segments = sorted(customer_model["RFM_Segment"].dropna().astype(str).unique())
    selected_segments = st.sidebar.multiselect("RFM Segment", segments, default=segments)
else:
    selected_segments = []

# Business priority filter
if "Business_Priority" in customer_model.columns:
    priorities = sorted(customer_model["Business_Priority"].dropna().astype(str).unique(), key=priority_number)
    selected_priorities = st.sidebar.multiselect("Business Priority", priorities, default=priorities)
else:
    selected_priorities = []

# Apply customer filters
customer_filtered = customer_model.copy()
if selected_segments:
    customer_filtered = customer_filtered[customer_filtered["RFM_Segment"].isin(selected_segments)]
if selected_priorities:
    customer_filtered = customer_filtered[customer_filtered["Business_Priority"].isin(selected_priorities)]

selected_customer_ids = set(customer_filtered["CustomerID"].astype(str))

# Apply sales filters
sales_filtered = sales_model.copy()
if selected_years and "Year" in sales_filtered.columns:
    sales_filtered = sales_filtered[sales_filtered["Year"].isin(selected_years)]
if selected_countries and "Country" in sales_filtered.columns:
    sales_filtered = sales_filtered[sales_filtered["Country"].isin(selected_countries)]
sales_filtered = sales_filtered[sales_filtered["CustomerID"].astype(str).isin(selected_customer_ids)]

# ------------------------------------------------------------
# Global calculations
# ------------------------------------------------------------
total_revenue = sales_filtered["Revenue"].sum()
total_transaction = sales_filtered["InvoiceNo"].nunique()
total_customer = sales_filtered["CustomerID"].nunique()
total_product = sales_filtered["ProductKey"].nunique()
avg_order_value = total_revenue / total_transaction if total_transaction else 0

# ------------------------------------------------------------
# PAGE 1 — Overview
# ------------------------------------------------------------
if page == "Overview":
    st.title("📊 Customer Intelligence Dashboard")
    st.caption("Project 1 — Online Retail Customer Intelligence untuk MBKM Studi Independen")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Total Revenue", rupiah_like(total_revenue))
    with c2: metric_card("Transactions", int_like(total_transaction))
    with c3: metric_card("Customers", int_like(total_customer))
    with c4: metric_card("Products", int_like(total_product))
    with c5: metric_card("Avg Order Value", rupiah_like(avg_order_value))

    st.markdown("---")
    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("Ringkasan Project")
        st.markdown(
            """
            Dashboard ini menampilkan hasil **Customer Intelligence** dari data transaksi Online Retail. 
            Analisis utama mencakup KPI penjualan, RFM Analysis, segmentasi pelanggan, clustering pendukung, 
            serta prioritas rekomendasi bisnis.
            """
        )
        st.markdown(
            """
            <div class="note-box">
            Streamlit digunakan sebagai reporting tool interaktif berbasis Python. 
            Fokus dashboard ini bukan hanya visualisasi, tetapi juga mendukung pengambilan keputusan bisnis.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.subheader("Output Utama")
        st.write("- Sales Performance")
        st.write("- Customer Segmentation")
        st.write("- Business Recommendation")
        st.write("- Data Model & Methodology")

    st.markdown("---")
    st.subheader("Insight Otomatis")
    if data["insight"]:
        st.text(data["insight"])
    else:
        st.info("File insight_project1.txt belum ditemukan. Dashboard tetap bisa berjalan.")

# ------------------------------------------------------------
# PAGE 2 — Sales Performance
# ------------------------------------------------------------
elif page == "Sales Performance":
    st.title("📈 Sales Performance")
    st.caption("KPI, tren revenue, negara, produk, dan customer dengan kontribusi tertinggi.")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Revenue", rupiah_like(total_revenue))
    with c2: st.metric("Transactions", int_like(total_transaction))
    with c3: st.metric("Customers", int_like(total_customer))
    with c4: st.metric("AOV", rupiah_like(avg_order_value))

    st.markdown("---")

    if "MonthYear" in sales_filtered.columns:
        monthly = (
            sales_filtered.groupby(["MonthYear"], as_index=False)
            .agg(Total_Revenue=("Revenue", "sum"), Total_Transaction=("InvoiceNo", "nunique"))
        )
        # Sort MonthYear aman karena format YYYY-MM
        monthly = monthly.sort_values("MonthYear")
        fig = px.line(monthly, x="MonthYear", y="Total_Revenue", markers=True, title="Monthly Revenue Trend")
        fig.update_layout(xaxis_title="Month", yaxis_title="Revenue")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        country_rev = (
            sales_filtered.groupby("Country", as_index=False)["Revenue"]
            .sum()
            .sort_values("Revenue", ascending=False)
            .head(10)
        )
        fig = px.bar(country_rev, x="Country", y="Revenue", title="Top 10 Country by Revenue")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        product_rev = (
            sales_filtered.groupby("ProductName", as_index=False)
            .agg(Revenue=("Revenue", "sum"), Transaction=("InvoiceNo", "nunique"))
            .query("Transaction >= 50")
            .sort_values("Revenue", ascending=False)
            .head(10)
        )
        fig = px.bar(product_rev, x="Revenue", y="ProductName", orientation="h", title="Top 10 Consistent Product by Revenue")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Customer by Revenue")
    top_cust = (
        sales_filtered.groupby("CustomerID", as_index=False)
        .agg(Revenue=("Revenue", "sum"), Transaction=("InvoiceNo", "nunique"))
        .sort_values("Revenue", ascending=False)
        .head(10)
    )
    st.dataframe(top_cust, use_container_width=True)

# ------------------------------------------------------------
# PAGE 3 — Customer Segmentation
# ------------------------------------------------------------
elif page == "Customer Segmentation":
    st.title("👥 Customer Segmentation")
    st.caption("RFM Segment sebagai dasar strategi bisnis; clustering digunakan sebagai validasi pola data.")

    col1, col2 = st.columns(2)

    with col1:
        seg_count = (
            customer_filtered.groupby("RFM_Segment", as_index=False)
            .agg(Customers=("CustomerID", "count"))
            .sort_values("Customers", ascending=False)
        )
        fig = px.bar(seg_count, x="RFM_Segment", y="Customers", title="Jumlah Customer per RFM Segment")
        fig.update_layout(xaxis_title="RFM Segment", yaxis_title="Customers")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        seg_rev = (
            customer_filtered.groupby("RFM_Segment", as_index=False)
            .agg(Total_Monetary=("Monetary", "sum"))
            .sort_values("Total_Monetary", ascending=False)
        )
        fig = px.bar(seg_rev, x="RFM_Segment", y="Total_Monetary", title="Total Monetary per RFM Segment")
        fig.update_layout(xaxis_title="RFM Segment", yaxis_title="Monetary")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        cluster_dist = (
            customer_filtered.groupby("Cluster_Label", as_index=False)
            .agg(Customers=("CustomerID", "count"))
            .sort_values("Customers", ascending=False)
        )
        fig = px.pie(cluster_dist, names="Cluster_Label", values="Customers", hole=0.45, title="Cluster Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        matrix = pd.crosstab(customer_filtered["RFM_Segment"], customer_filtered["Cluster_Label"])
        st.subheader("RFM Segment vs Cluster")
        st.dataframe(matrix, use_container_width=True)

    st.markdown("---")
    st.subheader("Customer Detail")
    detail_cols = [
        "CustomerID", "RFM_Segment", "Cluster_Label", "Recency", "Frequency", "Monetary",
        "Business_Priority", "Recommended_Action"
    ]
    detail_cols = [c for c in detail_cols if c in customer_filtered.columns]
    st.dataframe(customer_filtered[detail_cols].sort_values("Monetary", ascending=False), use_container_width=True)

    st.markdown(
        """
        <div class="danger-box">
        Catatan: RFM Segment digunakan sebagai dasar rekomendasi bisnis. Cluster hanya menjadi pendukung karena clustering dapat terpengaruh nilai Monetary ekstrem.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# PAGE 4 — Business Recommendation
# ------------------------------------------------------------
elif page == "Business Recommendation":
    st.title("🎯 Business Recommendation")
    st.caption("Prioritas bisnis berdasarkan RFM Segment, Risk Score, Value Score, dan Business Score.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Filtered Customers", int_like(customer_filtered["CustomerID"].nunique()))
    with c2:
        if "Business_Score" in customer_filtered.columns:
            st.metric("Avg Business Score", rupiah_like(customer_filtered["Business_Score"].mean()))
    with c3:
        st.metric("Total Monetary", rupiah_like(customer_filtered["Monetary"].sum()))

    priority_order = sorted(customer_filtered["Business_Priority"].dropna().unique(), key=priority_number)

    col1, col2 = st.columns(2)
    with col1:
        pr_count = (
            customer_filtered.groupby("Business_Priority", as_index=False)
            .agg(Customers=("CustomerID", "count"))
        )
        pr_count["sort"] = pr_count["Business_Priority"].apply(priority_number)
        pr_count = pr_count.sort_values("sort")
        fig = px.bar(pr_count, x="Business_Priority", y="Customers", title="Customer per Business Priority")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        pr_rev = (
            customer_filtered.groupby("Business_Priority", as_index=False)
            .agg(Total_Monetary=("Monetary", "sum"))
        )
        pr_rev["sort"] = pr_rev["Business_Priority"].apply(priority_number)
        pr_rev = pr_rev.sort_values("sort")
        fig = px.bar(pr_rev, x="Business_Priority", y="Total_Monetary", title="Monetary per Business Priority")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recommended Action Summary")
    rec_summary = (
        customer_filtered.groupby(["Business_Priority", "RFM_Segment", "Recommended_Action"], as_index=False)
        .agg(
            Customers=("CustomerID", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
            Avg_Business_Score=("Business_Score", "mean"),
            Total_Monetary=("Monetary", "sum"),
        )
        .round(2)
    )
    rec_summary["sort"] = rec_summary["Business_Priority"].apply(priority_number)
    rec_summary = rec_summary.sort_values(["sort", "Total_Monetary"], ascending=[True, False]).drop(columns="sort")
    st.dataframe(rec_summary, use_container_width=True)

    st.markdown(
        """
        <div class="note-box">
        Interpretasi penting: Priority 1 bukan berarti revenue terbesar. Priority 1 berarti paling urgent karena pelanggan bernilai tinggi mulai pasif. 
        Priority 2 adalah Champions, yaitu sumber revenue utama yang harus dipertahankan.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# PAGE 5 — Data Model & Methodology
# ------------------------------------------------------------
elif page == "Data Model & Methodology":
    st.title("🧩 Data Model & Methodology")
    st.caption("Halaman ini menjadi pengganti bukti Model View Power BI: relasi data, workflow, metode, dan batasan analisis.")

    st.subheader("Star Schema")
    schema = pd.DataFrame({
        "Table": ["fact_sales", "dim_customer", "dim_product", "dim_country", "dim_date"],
        "Role": ["Fact table", "Customer dimension", "Product dimension", "Country dimension", "Date dimension"],
        "Key": ["CustomerID, ProductKey, CountryKey, DateKey", "CustomerID", "ProductKey", "CountryKey", "DateKey"],
        "Rows": [len(fact_sales), len(dim_customer), len(dim_product), len(dim_country), len(dim_date)],
    })
    st.dataframe(schema, use_container_width=True)

    relation = pd.DataFrame({
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
        ]
    })
    st.dataframe(relation, use_container_width=True)

    st.markdown("---")

    tabs = st.tabs(["BI Workflow", "Segmentation", "Business Modeling", "Method Decision", "Course Mapping", "Limitations"])

    with tabs[0]:
        if data["bi_workflow"] is not None:
            st.dataframe(data["bi_workflow"], use_container_width=True)
        else:
            st.info("bi_workflow.csv belum ditemukan.")

    with tabs[1]:
        if data["segmentation_workflow"] is not None:
            st.dataframe(data["segmentation_workflow"], use_container_width=True)
        else:
            st.info("segmentation_workflow.csv belum ditemukan.")

    with tabs[2]:
        if data["business_modeling_workflow"] is not None:
            st.dataframe(data["business_modeling_workflow"], use_container_width=True)
        else:
            st.info("business_modeling_workflow.csv belum ditemukan.")

    with tabs[3]:
        if data["method_decision"] is not None:
            st.dataframe(data["method_decision"], use_container_width=True)
        else:
            st.info("method_decision.csv belum ditemukan.")

    with tabs[4]:
        if data["course_mapping"] is not None:
            st.dataframe(data["course_mapping"], use_container_width=True)
        else:
            st.info("course_mapping_project1.csv belum ditemukan.")

    with tabs[5]:
        if data["limitations"]:
            st.text(data["limitations"])
        else:
            st.info("limitations_project1.txt belum ditemukan.")
