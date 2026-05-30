# Project 1 Streamlit Dashboard v2

Dashboard final untuk Project 1 Customer Intelligence Online Retail.

## Cara menjalankan

Pastikan struktur folder:

```text
project_dashboard/
├── app.py
├── requirements.txt
└── project1_output/
    ├── fact_sales.csv
    ├── dim_customer.csv
    ├── dim_product.csv
    ├── dim_country.csv
    ├── dim_date.csv
    └── file output lain dari Colab
```

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

## Halaman

1. Overview
2. Sales Performance
3. Customer Segmentation
4. Business Recommendation
5. Data Model & Methodology

## Catatan

- Dashboard ini memakai Streamlit sebagai reporting tool berbasis Python.
- RFM Segment menjadi dasar rekomendasi bisnis.
- Clustering hanya menjadi pendukung.
- Top Product sudah mengecualikan non-product items seperti POSTAGE dan MANUAL.
