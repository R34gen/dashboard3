from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Project 1 | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLE
# ============================================================
st.markdown("""
<style>
.main .block-container {padding-top: 2rem; padding-bottom: 4rem; max-width: 1320px;}
h1, h2, h3 {letter-spacing: -0.04em;}
.subtitle {color:#9ca3af; font-size:1.03rem; margin-top:-0.45rem; margin-bottom:1rem;}
.kpi-card {
    background: linear-gradient(135deg, #ffffff 0%, #f5f7fb 100%);
    color:#111827; border-radius:18px; padding:20px 22px;
    border:1px solid rgba(255,255,255,0.08); min-height:112px;
    box-shadow:0 14px 34px rgba(0,0,0,0.16);
}
.kpi-label {color:#6b7280; font-size:0.92rem; font-weight:700; margin-bottom:10px;}
.kpi-value {color:#111827; font-size:1.85rem; font-weight:850; line-height:1.15;}
.info-box, .warn-box, .good-box {
    color:#f9fafb; padding:16px 20px; border-radius:14px;
    margin-top:12px; margin-bottom:16px; line-height:1.65;
}
.info-box {background:#1f2937; border-left:6px solid #7cc4ff;}
.warn-box {background:#2a1f10; color:#ffedd5; border-left:6px solid #f97316;}
.good-box {background:#102a1f; color:#d1fae5; border-left:6px solid #22c55e;}
hr {border:none; height:1px; background:rgba(255,255,255,0.12); margin:1.5rem 0;}
div[data-testid="stDataFrame"] {border-radius:14px; overflow:hidden;}
</style>
""", unsafe_allow_html=True)

SEGMENT_ORDER = ["Champions", "Loyal Customers", "Big Spenders", "High Value At Risk", "At Risk", "Regular Customers", "Lost Customers"]
PRIORITY_ORDER = ["Priority 1", "Priority 2", "Priority 3", "Priority 4", "Priority 5", "Priority 6"]
NON_PRODUCT = ["POSTAGE", "DOTCOM POSTAGE", "CARRIAGE", "MANUAL", "BANK CHARGES", "AMAZON FEE", "SAMPLES", "ADJUST", "ADJUSTMENT", "CRUK COMMISSION"]
PLOTLY_TEMPLATE = "plotly_dark"
PRIMARY_COLOR = "#7cc4ff"

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

def kpi_card(label, value):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def box(kind, html):
    cls = {"info": "info-box", "warn": "warn-box", "good": "good-box"}.get(kind, "info-box")
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

def clean_text(df):
    if df is None:
        return None
    out = df.copy()
    for c in out.select_dtypes(include="object").columns:
        out[c] = out[c].astype(str).str.strip()
    return out

def safe_num(df, cols):
    if df is None:
        return df
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def rename_display(df):
    mapping = {
        "CustomerID":"Customer ID", "RFM_Segment":"RFM Segment", "Cluster_Label":"Cluster Label",
        "Business_Priority":"Business Priority", "Recommended_Action":"Recommended Action",
        "Risk_Score":"Risk Score", "Value_Score":"Value Score", "Business_Score":"Business Score",
        "Total_Monetary":"Total Monetary", "Avg_Business_Score":"Avg Business Score",
        "Avg_Recency":"Avg Recency", "Avg_Frequency":"Avg Frequency", "Avg_Monetary":"Avg Monetary",
        "ProductName":"Product Name", "ProductKey":"Product Key", "CountryKey":"Country Key", "DateKey":"Date Key",
    }
    return df.rename(columns={k:v for k,v in mapping.items() if k in df.columns})

def prepare_sales_model(fact_sales, dim_customer, dim_product, dim_country, dim_date):
    sales = fact_sales.copy()
    if dim_product is not None and {"ProductKey","ProductName"}.issubset(dim_product.columns):
        sales = sales.merge(dim_product[["ProductKey","ProductName"]], on="ProductKey", how="left")
    if dim_country is not None and {"CountryKey","Country"}.issubset(dim_country.columns):
        sales = sales.merge(dim_country[["CountryKey","Country"]], on="CountryKey", how="left")
    if dim_date is not None and "DateKey" in dim_date.columns:
        date_cols = [c for c in ["DateKey","Date","Year","Month","MonthName","Quarter","MonthYear"] if c in dim_date.columns]
        sales = sales.merge(dim_date[date_cols], on="DateKey", how="left")
    if dim_customer is not None and "CustomerID" in dim_customer.columns:
        cust_cols = [c for c in ["CustomerID","RFM_Segment","Cluster_Label","Business_Priority","Recommended_Action","Business_Score"] if c in dim_customer.columns]
        sales = sales.merge(dim_customer[cust_cols], on="CustomerID", how="left")
    return sales

def bar(df, x, y, title, order=None):
    fig = px.bar(df, x=x, y=y, title=title, template=PLOTLY_TEMPLATE, color_discrete_sequence=[PRIMARY_COLOR])
    if order:
        fig.update_xaxes(categoryorder="array", categoryarray=order)
    fig.update_layout(height=390, title_font_size=18, margin=dict(l=20,r=20,t=55,b=35), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def hbar(df, x, y, title):
    fig = px.bar(df, x=x, y=y, orientation="h", title=title, template=PLOTLY_TEMPLATE, color_discrete_sequence=[PRIMARY_COLOR])
    fig.update_layout(height=420, title_font_size=18, margin=dict(l=20,r=20,t=55,b=35), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(autorange="reversed"))
    return fig

def remove_non_product(df, col):
    if col not in df.columns:
        return df
    return df[~df[col].astype(str).str.upper().str.strip().isin(NON_PRODUCT)].copy()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Project 1")
st.sidebar.caption("Customer Intelligence Online Retail")

data_dir = st.sidebar.text_input("Data folder", value="project1_output")

page = st.sidebar.radio(
    "Navigasi",
    [
        "1. Executive Overview",
        "2. Sales & BI",
        "3. Customer Segmentation",
        "4. Business Recommendation",
        "5. MBKM Evidence",
    ],
)

# ============================================================
# LOAD DATA
# ============================================================
fact_sales = clean_text(load_csv("fact_sales.csv", data_dir))
dim_customer = clean_text(load_csv("dim_customer.csv", data_dir))
dim_product = clean_text(load_csv("dim_product.csv", data_dir))
dim_country = clean_text(load_csv("dim_country.csv", data_dir))
dim_date = clean_text(load_csv("dim_date.csv", data_dir))
business_summary = clean_text(load_csv("business_summary.csv", data_dir))
rfm_cluster_matrix = clean_text(load_csv("rfm_cluster_matrix.csv", data_dir))
bi_workflow = clean_text(load_csv("bi_workflow.csv", data_dir))
segmentation_workflow = clean_text(load_csv("segmentation_workflow.csv", data_dir))
business_modeling_workflow = clean_text(load_csv("business_modeling_workflow.csv", data_dir))
method_decision = clean_text(load_csv("method_decision.csv", data_dir))
course_mapping = clean_text(load_csv("course_mapping_project1.csv", data_dir))

if fact_sales is None or dim_customer is None:
    st.error("File utama belum ditemukan. Pastikan project1_output berisi fact_sales.csv dan dim_customer.csv.")
    st.stop()

fact_sales = safe_num(fact_sales, ["Revenue","Quantity","UnitPrice","DateKey","ProductKey","CountryKey"])
dim_customer = safe_num(dim_customer, ["Monetary","Recency","Frequency","Risk_Score","Value_Score","Business_Score"])
dim_date = safe_num(dim_date, ["DateKey","Year","Month"]) if dim_date is not None else None

sales_model = prepare_sales_model(fact_sales, dim_customer, dim_product, dim_country, dim_date)

# Filters
st.sidebar.markdown("---")
st.sidebar.subheader("Filter")
st.sidebar.caption("Kosongkan = semua data.")

years = sorted([int(x) for x in sales_model["Year"].dropna().unique()]) if "Year" in sales_model.columns else []
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

total_revenue = filtered_sales["Revenue"].sum() if "Revenue" in filtered_sales.columns else 0
total_transaction = filtered_sales["InvoiceNo"].nunique() if "InvoiceNo" in filtered_sales.columns else 0
total_customer = filtered_sales["CustomerID"].nunique() if "CustomerID" in filtered_sales.columns else 0
total_product = filtered_sales["ProductKey"].nunique() if "ProductKey" in filtered_sales.columns else 0
aov = total_revenue / total_transaction if total_transaction else 0

# ============================================================
# PAGE 1
# ============================================================
if page == "1. Executive Overview":
    st.title("📊 Executive Overview")
    st.markdown('<div class="subtitle">Ringkasan Project 1: Customer Intelligence Online Retail.</div>', unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: kpi_card("Total Revenue", money_fmt(total_revenue))
    with c2: kpi_card("Transactions", int_fmt(total_transaction))
    with c3: kpi_card("Customers", int_fmt(total_customer))
    with c4: kpi_card("Products", int_fmt(total_product))
    with c5: kpi_card("AOV", money_fmt(aov))

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.15, 1])
    with col1:
        st.subheader("Key Findings")
        box("info", """
        <b>1. Champions adalah revenue driver.</b><br>
        Pelanggan terbaik harus dipertahankan dengan loyalty program.
        <br><br>
        <b>2. High Value At Risk paling urgent.</b><br>
        Mereka bernilai tinggi tetapi mulai pasif, jadi perlu reaktivasi personal.
        <br><br>
        <b>3. Lost Customers jangan diberi campaign mahal.</b><br>
        Jumlahnya besar, tetapi harus ditangani dengan win-back campaign biaya rendah.
        """)
    with col2:
        st.subheader("Batas Klaim")
        box("warn", """
        Segmentasi yang diklaim adalah <b>perilaku transaksi berbasis RFM</b>, bukan demografis.
        Business priority adalah <b>rule-based decision model</b>, bukan prediksi churn.
        Revenue bukan profit karena dataset tidak memiliki biaya/margin.
        """)

    st.subheader("BAB 4 Mapping Singkat")
    display_df(pd.DataFrame({
        "Subbab": ["4.1", "4.2", "4.3", "4.4", "4.5"],
        "Mata Kuliah": ["Softskill / Kepemimpinan", "Kecerdasan Bisnis", "Analisis Segmentasi Pelanggan", "Pemodelan Data Bisnis", "Intuisi dan Wawasan Data"],
        "Bukti Utama": ["logbook, method decision", "ETL, star schema, KPI", "RFM, cluster pendukung", "business score, priority, action", "insight, limitations"],
    }))

# ============================================================
# PAGE 2
# ============================================================
elif page == "2. Sales & BI":
    st.title("📈 Sales & Business Intelligence")
    st.markdown('<div class="subtitle">Bukti BI: KPI, reporting, visualisasi, star schema, dan workflow.</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi_card("Total Revenue", money_fmt(total_revenue))
    with c2: kpi_card("Transactions", int_fmt(total_transaction))
    with c3: kpi_card("Customers", int_fmt(total_customer))
    with c4: kpi_card("AOV", money_fmt(aov))

    if "Country" in filtered_sales.columns and len(filtered_sales) > 0:
        top_country = filtered_sales.groupby("Country")["Revenue"].sum().sort_values(ascending=False).index[0]
        box("info", f"Halaman ini membuktikan <b>Kecerdasan Bisnis</b>: KPI, tren revenue, reporting, dan visualisasi. Negara dominan pada filter aktif adalah <b>{top_country}</b>.")

    if {"MonthYear","Revenue"}.issubset(filtered_sales.columns):
        monthly = filtered_sales.groupby(["Year","Month","MonthYear"], as_index=False).agg(Revenue=("Revenue","sum")).sort_values(["Year","Month"])
        fig = px.line(monthly, x="MonthYear", y="Revenue", markers=True, title="Monthly Revenue Trend", template=PLOTLY_TEMPLATE, color_discrete_sequence=[PRIMARY_COLOR])
        fig.update_layout(height=400, margin=dict(l=20,r=20,t=55,b=35), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        box("warn", "Data Desember 2011 tidak penuh satu bulan, jadi tidak boleh dibandingkan mentah sebagai performa bulanan penuh.")

    col1,col2 = st.columns(2)
    with col1:
        if "Country" in filtered_sales.columns:
            country_chart = filtered_sales.groupby("Country", as_index=False).agg(Revenue=("Revenue","sum")).sort_values("Revenue", ascending=False).head(10)
            st.plotly_chart(bar(country_chart, "Country", "Revenue", "Top 10 Country by Revenue"), use_container_width=True)
    with col2:
        if "ProductName" in filtered_sales.columns:
            prod = remove_non_product(filtered_sales, "ProductName")
            product_chart = prod.groupby("ProductName", as_index=False).agg(Revenue=("Revenue","sum"), Transaction=("InvoiceNo","nunique")).query("Transaction >= 50").sort_values("Revenue", ascending=False).head(10)
            st.plotly_chart(hbar(product_chart, "Revenue", "ProductName", "Top 10 Consistent Product by Revenue"), use_container_width=True)

    with st.expander("Star Schema dan Relationship", expanded=False):
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

# ============================================================
# PAGE 3
# ============================================================
elif page == "3. Customer Segmentation":
    st.title("👥 Customer Segmentation")
    st.markdown('<div class="subtitle">Bukti segmentasi pelanggan: RFM sebagai dasar utama, cluster sebagai pendukung.</div>', unsafe_allow_html=True)

    box("warn", "Jangan klaim segmentasi demografis. Dataset tidak punya usia, gender, pekerjaan, pendapatan, atau psikografis.")

    c1,c2,c3 = st.columns(3)
    with c1: box("good", "<h4>Champions</h4>Pelanggan terbaik. Pertahankan dengan loyalty program.")
    with c2: box("warn", "<h4>High Value At Risk</h4>Bernilai tinggi tetapi mulai pasif. Target reaktivasi.")
    with c3: box("info", "<h4>Regular & Lost</h4>Gunakan engagement dan win-back campaign hemat biaya.")

    col1,col2 = st.columns(2)
    if "RFM_Segment" in filtered_customer.columns:
        seg_count = filtered_customer.groupby("RFM_Segment", as_index=False).agg(Customers=("CustomerID","count"))
        seg_count["RFM_Segment"] = pd.Categorical(seg_count["RFM_Segment"], categories=SEGMENT_ORDER, ordered=True)
        seg_count = seg_count.sort_values("RFM_Segment")
        with col1:
            st.plotly_chart(bar(seg_count, "RFM_Segment", "Customers", "Customer per RFM Segment", SEGMENT_ORDER), use_container_width=True)

    if {"RFM_Segment","Monetary"}.issubset(filtered_customer.columns):
        seg_rev = filtered_customer.groupby("RFM_Segment", as_index=False).agg(Monetary=("Monetary","sum"))
        seg_rev["RFM_Segment"] = pd.Categorical(seg_rev["RFM_Segment"], categories=SEGMENT_ORDER, ordered=True)
        seg_rev = seg_rev.sort_values("RFM_Segment")
        with col2:
            st.plotly_chart(bar(seg_rev, "RFM_Segment", "Monetary", "Monetary per RFM Segment", SEGMENT_ORDER), use_container_width=True)

    col1,col2 = st.columns([0.9,1.1])
    with col1:
        if "Cluster_Label" in filtered_customer.columns:
            cluster_dist = filtered_customer.groupby("Cluster_Label", as_index=False).agg(Customers=("CustomerID","count")).sort_values("Customers", ascending=False)
            fig = px.pie(cluster_dist, names="Cluster_Label", values="Customers", hole=0.55, title="Cluster Distribution", template=PLOTLY_TEMPLATE)
            fig.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("RFM vs Cluster")
        if rfm_cluster_matrix is not None:
            display_df(rename_display(rfm_cluster_matrix), height=400)

    with st.expander("Customer Detail", expanded=False):
        cols = [c for c in ["CustomerID","RFM_Segment","Cluster_Label","Recency","Frequency","Monetary","Business_Priority","Recommended_Action"] if c in filtered_customer.columns]
        detail = filtered_customer[cols].copy()
        if "Monetary" in detail.columns:
            detail = detail.sort_values("Monetary", ascending=False)
        display_df(rename_display(detail.head(30)), height=420)

# ============================================================
# PAGE 4
# ============================================================
elif page == "4. Business Recommendation":
    st.title("🎯 Business Recommendation")
    st.markdown('<div class="subtitle">Bukti pemodelan data bisnis: risk score, value score, business score, priority, action.</div>', unsafe_allow_html=True)

    box("warn", "Ini bukan prediksi churn. Ini rule-based business decision model. Recency dipakai sebagai proxy risiko pasif, Monetary sebagai proxy customer value.")

    avg_score = filtered_customer["Business_Score"].mean() if "Business_Score" in filtered_customer.columns else 0
    total_monetary = filtered_customer["Monetary"].sum() if "Monetary" in filtered_customer.columns else 0
    c1,c2,c3 = st.columns(3)
    with c1: kpi_card("Customers", int_fmt(filtered_customer["CustomerID"].nunique()))
    with c2: kpi_card("Avg Business Score", money_fmt(avg_score))
    with c3: kpi_card("Total Monetary", money_fmt(total_monetary))

    a1,a2,a3 = st.columns(3)
    with a1: box("warn", "<h4>Priority 1</h4>Reaktivasi personal untuk High Value At Risk.")
    with a2: box("good", "<h4>Priority 2</h4>Pertahankan Champions dengan loyalty program.")
    with a3: box("info", "<h4>Priority 3</h4>Cross-selling dan bundling untuk Big Spenders/Loyal.")

    if "Business_Priority" in filtered_customer.columns:
        prio = filtered_customer.groupby("Business_Priority", as_index=False).agg(Customers=("CustomerID","count"), Total_Monetary=("Monetary","sum"), Avg_Business_Score=("Business_Score","mean")).round(2)
        prio["Business_Priority"] = pd.Categorical(prio["Business_Priority"], categories=PRIORITY_ORDER, ordered=True)
        prio = prio.sort_values("Business_Priority")
        col1,col2 = st.columns(2)
        with col1:
            st.plotly_chart(bar(prio, "Business_Priority", "Customers", "Customer per Priority", PRIORITY_ORDER), use_container_width=True)
        with col2:
            st.plotly_chart(bar(prio, "Business_Priority", "Total_Monetary", "Monetary per Priority", PRIORITY_ORDER), use_container_width=True)

    box("warn", "Priority 1 bukan revenue terbesar. Priority 1 berarti paling urgent karena pelanggan bernilai tinggi mulai pasif.")

    if business_summary is not None:
        with st.expander("Recommended Action Summary", expanded=False):
            display_df(rename_display(business_summary), height=380)

# ============================================================
# PAGE 5
# ============================================================
elif page == "5. MBKM Evidence":
    st.title("🧩 MBKM Evidence")
    st.markdown('<div class="subtitle">Halaman ini untuk membantu penjelasan BAB 4, bukan untuk dibaca seperti dashboard bisnis.</div>', unsafe_allow_html=True)

    evidence = pd.DataFrame({
        "Mata Kuliah": ["Softskill / Kepemimpinan", "Kecerdasan Bisnis", "Analisis Segmentasi Pelanggan", "Pemodelan Data Bisnis", "Intuisi dan Wawasan Data"],
        "Bukti Dashboard": [
            "Method decision, batasan, bukti keputusan; tetap perlu logbook/konsultasi",
            "KPI, ETL, star schema, sales visualization",
            "RFM segment, cluster distribution, RFM vs cluster",
            "Risk/Value/Business Score, priority, recommended action",
            "Executive summary, key findings, limitations, rekomendasi",
        ],
        "Screenshot Utama": [
            "MBKM Evidence + method decision",
            "Sales & BI + star schema",
            "Customer Segmentation",
            "Business Recommendation",
            "Executive Overview",
        ],
    })
    display_df(evidence)

    st.markdown("<hr>", unsafe_allow_html=True)

    for title, df in [
        ("BI Workflow", bi_workflow),
        ("Segmentation Workflow", segmentation_workflow),
        ("Business Modeling Workflow", business_modeling_workflow),
        ("Method Decision", method_decision),
        ("Course Mapping", course_mapping),
    ]:
        with st.expander(title, expanded=False):
            if df is not None:
                if title == "BI Workflow":
                    df = df.replace("Power BI", "Streamlit", regex=True)
                display_df(rename_display(df), height=360)
            else:
                st.caption(f"{title} tidak ditemukan.")

    with st.expander("Limitations", expanded=False):
        limitations_path = Path(data_dir) / "limitations_project1.txt"
        if limitations_path.exists():
            st.text(limitations_path.read_text(encoding="utf-8"))
        else:
            st.write("limitations_project1.txt tidak ditemukan.")

    box("info", "Untuk laporan, jangan screenshot browser penuh. Crop area dashboard saja: tanpa address bar, tanpa sidebar kalau tidak dibutuhkan, tanpa Manage App.")