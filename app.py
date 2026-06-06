import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import os
from datetime import datetime

# Import preprocessing engine
from preprocess import clean_dataframe, detect_column_types, clean_generic_dataframe

# Set page configuration
st.set_page_config(
    page_title="Data Cleaning & Visualization Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #94A3B8;
        margin-bottom: 2rem;
    }
    .section-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: 700;
        color: #6366F1;
        margin-bottom: 2px;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load raw sample data
@st.cache_data
def get_raw_sample_data():
    raw_path = "data/raw_sales_data.csv"
    if os.path.exists(raw_path):
        return pd.read_csv(raw_path)
    return None

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.image("https://img.icons8.com/nolan/128/bar-chart.png", width=60)
st.sidebar.markdown("<h2 style='color:#6366F1;'>Studio Menu</h2>", unsafe_allow_html=True)

# 1. Dataset Source Selection
dataset_source = st.sidebar.radio(
    "Choose Dataset Source:",
    ["📦 Sample E-commerce Sales Dataset", "📥 Upload Custom CSV File"]
)

# Initialize variables
df_raw = None
df_clean = None
col_types = None
change_log = None
is_custom_dataset = False

if dataset_source == "📦 Sample E-commerce Sales Dataset":
    df_raw = get_raw_sample_data()
    if df_raw is not None:
        # For the sample, we use the specialized cleaning rules
        df_clean, sample_stats = clean_dataframe(df_raw)
        # Re-map sample stats to matches change_log formatting
        change_log = {
            "duplicates_removed": sample_stats.get("duplicates_removed", 0),
            "Date": {"imputed_missing": f"Fixed {sample_stats.get('missing_dates_found', 0)} nulls & {sample_stats.get('future_dates_corrected', 0)} future values"},
            "Customer_Gender": {"imputed_missing": f"Imputed {sample_stats.get('missing_genders_imputed', 0)} records"},
            "Product_Category": {"imputed_missing": f"Imputed {sample_stats.get('missing_categories_imputed', 0)} records"},
            "Customer_Age": {"imputed_missing": f"Corrected {sample_stats.get('invalid_ages_corrected', 0)} invalid/null ages"},
            "Quantity": {"outliers_clamped": f"Clamped {sample_stats.get('quantity_outliers_clamped', 0)} outliers"},
            "Unit_Price": {"outliers_clamped": f"Clamped {sample_stats.get('price_outliers_clamped', 0)} outliers"},
            "Total_Amount": {"imputed_missing": f"Recalculated {sample_stats.get('total_amount_mismatches_fixed', 0)} mismatched values"},
            "Payment_Method": {"imputed_missing": f"Imputed {sample_stats.get('missing_payments_imputed', 0)} records"},
            "Customer_Rating": {"imputed_missing": f"Corrected {sample_stats.get('invalid_ratings_corrected', 0)} ratings"}
        }
        col_types = detect_column_types(df_raw)
else:
    is_custom_dataset = True
    uploaded_file = st.sidebar.file_uploader("Upload CSV:", type=["csv"])
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            col_types = detect_column_types(df_raw)
            
            # Show Cleaning Configurator
            st.sidebar.markdown("### Cleaning Parameters")
            drop_dups = st.sidebar.checkbox("Remove Duplicates", value=True)
            
            st.sidebar.markdown("**Numerical columns:**")
            num_impute = st.sidebar.selectbox("Imputation Strategy", ["median", "mean", "mode", "zero", "none"], key="num_imp")
            clamp_outliers = st.sidebar.checkbox("Clamp Outliers (IQR)", value=True)
            iqr_k = st.sidebar.slider("IQR Multiplier (k)", 1.0, 5.0, 1.5, 0.1)
            
            st.sidebar.markdown("**Categorical columns:**")
            cat_impute = st.sidebar.selectbox("Imputation Strategy", ["placeholder", "mode", "none"], key="cat_imp")
            cat_placeholder = st.sidebar.text_input("Placeholder string", value="Unknown")
            cat_case = st.sidebar.selectbox("Text Casing", ["none", "Title Case", "lowercase", "UPPERCASE"])
            
            st.sidebar.markdown("**Datetime columns:**")
            dt_impute = st.sidebar.selectbox("Imputation Strategy", ["median", "now", "none"], key="dt_imp")
            
            # Compile config
            config = {
                "drop_duplicates": drop_dups,
                "numerical": {
                    "impute_strategy": num_impute,
                    "clamp_outliers": clamp_outliers,
                    "outlier_iqr_k": iqr_k
                },
                "categorical": {
                    "impute_strategy": cat_impute,
                    "placeholder_value": cat_placeholder,
                    "normalize_case": cat_case,
                    "strip_whitespace": True
                },
                "datetime": {
                    "impute_strategy": dt_impute
                }
            }
            
            df_clean, col_types, change_log = clean_generic_dataframe(df_raw, config)
        except Exception as e:
            st.sidebar.error(f"Error reading file: {str(e)}")

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select Studio View:",
    ["👋 Welcome & Overview", "🔍 Data Cleaning Explorer", "📈 Insights & Storytelling"]
)

# ==========================================
# PAGE 1: Welcome & Overview
# ==========================================
if page == "👋 Welcome & Overview":
    st.markdown("<div class='main-header'>Data Cleaning & Visualization Studio</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Clean arbitrary datasets, configure customized cleaning pipelines, and view visual reports.</div>", unsafe_allow_html=True)

    if df_raw is None:
        st.info("💡 To begin, upload a custom CSV file via the sidebar, or explore the preloaded E-commerce dataset.")
        
        # Guide uploader UI
        st.markdown("""
        ### What this application does:
        1. **Detects Column Types**: Ingests files and automatically classifies fields into Numerical, Categorical, or Datetime types.
        2. **Executes Configurable Cleaning Rules**: Handles duplicates, formats date strings, normalizes text casing, and cleans numerical values via your sidebar preferences.
        3. **Renders Dynamic Dashboards**: Builds visualization panels (correlation heatmaps, scatter plots, trend graphs) that adapt automatically to whatever variables exist in your upload.
        """)
        
        # Display sample dashboard metrics placeholder
        sample_df = get_raw_sample_data()
        if sample_df is not None:
            st.markdown("---")
            st.markdown("#### Sample E-commerce Dataset Structure (Out-of-the-Box Demo):")
            col1, col2, col3 = st.columns(3)
            col1.metric("Sample Rows", f"{len(sample_df):,}")
            col2.metric("Sample Columns", len(sample_df.columns))
            col3.metric("Sample Null Cells", f"{sample_df.isnull().sum().sum():,}")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Dataset Properties")
            if is_custom_dataset:
                st.success("📂 Successfully loaded your custom dataset!")
            else:
                st.info("📦 Visualizing the E-commerce transaction dataset.")
                
            st.markdown(f"""
            - **Total Rows:** {len(df_raw):,}
            - **Total Columns:** {len(df_raw.columns)}
            - **Duplicated Row Count:** {df_raw.duplicated().sum()}
            - **Missing Value Count (Total):** {df_raw.isnull().sum().sum():,}
            """)
            
            # Show inferred column types
            st.markdown("### Inferred Column Types")
            c_num, c_cat, c_dt = st.columns(3)
            
            with c_num:
                st.markdown("**🔢 Numerical Columns**")
                for col in col_types["numerical"]:
                    st.write(f"- `{col}`")
                if not col_types["numerical"]:
                    st.write("*None*")
                    
            with c_cat:
                st.markdown("**🔤 Categorical / Text Columns**")
                for col in col_types["categorical"]:
                    st.write(f"- `{col}`")
                if not col_types["categorical"]:
                    st.write("*None*")
                    
            with c_dt:
                st.markdown("**📅 Datetime Columns**")
                for col in col_types["datetime"]:
                    st.write(f"- `{col}`")
                if not col_types["datetime"]:
                    st.write("*None*")
                    
        with col2:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.subheader("Data Summary Metrics")
            st.metric("Total Cells", f"{df_raw.size:,}")
            st.metric("Completeness Rate", f"{round((1 - df_raw.isnull().sum().sum() / df_raw.size) * 100, 2)}%")
            st.metric("Unique Records Ratio", f"{round((len(df_raw.drop_duplicates()) / len(df_raw)) * 100, 1)}%")
            st.markdown("</div>", unsafe_allow_html=True)

            # Simple download button for cleaned dataset
            if df_clean is not None:
                csv_buffer = io.StringIO()
                df_clean.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Cleaned Dataset",
                    data=csv_buffer.getvalue(),
                    file_name="cleaned_dataset.csv",
                    mime="text/csv",
                    width="stretch"
                )

# ==========================================
# PAGE 2: Data Cleaning Explorer
# ==========================================
elif page == "🔍 Data Cleaning Explorer":
    st.markdown("<div class='main-header'>Data Cleaning Explorer</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Inspect the pipeline operations and view raw vs cleaned comparisons.</div>", unsafe_allow_html=True)

    if df_raw is None or df_clean is None:
        st.warning("⚠️ No active dataset. Please upload a dataset or select the sample from the sidebar.")
    else:
        # Show overall metrics
        st.markdown("### Preprocessing Pipeline Actions")
        
        # Display duplicates removed
        dups_removed = change_log.get("duplicates_removed", 0)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f"<div class='section-card'><div class='metric-val'>{dups_removed}</div><div class='metric-label'>Duplicates Deleted</div></div>", unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"<div class='section-card'><div class='metric-val'>{df_raw.isnull().sum().sum() - df_clean.isnull().sum().sum()}</div><div class='metric-label'>Missing Values Resolved</div></div>", unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"<div class='section-card'><div class='metric-val'>{len(df_clean)}</div><div class='metric-label'>Final Record Count</div></div>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📋 Data View & Log", "📊 Outlier Distribution", "📝 Categorical Comparison"])
        
        with tab1:
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("**Raw Data Sample**")
                st.dataframe(df_raw.head(10), height=250)
            with col_r:
                st.markdown("**Cleaned Data Sample**")
                st.dataframe(df_clean.head(10), height=250)
                
            # Render Preprocessing Change Log
            st.subheader("Transformation Log (by Column)")
            log_entries = []
            for col, logs in change_log.items():
                if col == "duplicates_removed":
                    continue
                for action, desc in logs.items():
                    log_entries.append({"Column": col, "Correction Action": action.replace('_', ' ').title(), "Details": desc})
                    
            if log_entries:
                st.table(pd.DataFrame(log_entries))
            else:
                st.info("No modifications were required based on your cleaning parameters.")

        with tab2:
            st.subheader("Outlier Distribution (IQR Method)")
            
            numeric_cols = col_types["numerical"]
            if not numeric_cols:
                st.info("No numerical columns detected to analyze outliers.")
            else:
                outlier_col = st.selectbox("Select numerical column to analyze:", numeric_cols)
                
                fig_box_raw = px.box(df_raw, y=outlier_col, title=f"Raw {outlier_col} (With Outliers)", color_discrete_sequence=["#EF4444"])
                fig_box_clean = px.box(df_clean, y=outlier_col, title=f"Cleaned {outlier_col} (Capped/Imputed)", color_discrete_sequence=["#10B981"])
                
                col_box1, col_box2 = st.columns(2)
                with col_box1:
                    fig_box_raw.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
                    st.plotly_chart(fig_box_raw, width="stretch")
                with col_box2:
                    fig_box_clean.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
                    st.plotly_chart(fig_box_clean, width="stretch")

        with tab3:
            st.subheader("Categorical Casing & Null Compactions")
            
            categorical_cols = col_types["categorical"]
            if not categorical_cols:
                st.info("No categorical columns detected to compare.")
            else:
                cat_col = st.selectbox("Select categorical column to compare:", categorical_cols)
                
                raw_c = df_raw[cat_col].fillna("Missing").value_counts().reset_index()
                raw_c.columns = ["Category", "Count"]
                raw_c["Dataset"] = "Raw"
                
                clean_c = df_clean[cat_col].value_counts().reset_index()
                clean_c.columns = ["Category", "Count"]
                clean_c["Dataset"] = "Cleaned"
                
                cat_compare = pd.concat([raw_c, clean_c])
                
                # Limit categories plotted to top 20 to avoid cluttered labels
                top_categories = df_clean[cat_col].value_counts().head(20).index
                cat_compare_filtered = cat_compare[cat_compare["Category"].isin(top_categories) | (cat_compare["Category"] == "Missing")]
                
                fig_cat = px.bar(
                    cat_compare_filtered, 
                    x="Category", 
                    y="Count", 
                    color="Dataset", 
                    barmode="group",
                    title=f"Category Frequencies (Top 20 Categories for '{cat_col}')",
                    color_discrete_sequence=["#F59E0B", "#6366F1"]
                )
                fig_cat.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
                st.plotly_chart(fig_cat, width="stretch")

# ==========================================
# PAGE 3: Insights & Storytelling
# ==========================================
elif page == "📈 Insights & Storytelling":
    st.markdown("<div class='main-header'>Insights & Storytelling</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Uncover analytical structures and correlations dynamically generated from the dataset.</div>", unsafe_allow_html=True)

    if df_raw is None or df_clean is None:
        st.warning("⚠️ No active dataset. Please upload a dataset or select the sample from the sidebar.")
    else:
        # Create tabs dynamically
        has_numeric = len(col_types["numerical"]) > 0
        has_categorical = len(col_types["categorical"]) > 0
        has_datetime = len(col_types["datetime"]) > 0
        
        tab_list = []
        if has_numeric:
            tab_list.append("🔗 Variable Relationships")
        if has_categorical and has_numeric:
            tab_list.append("📊 Segment Aggregations")
        if has_datetime and has_numeric:
            tab_list.append("📈 Chronological Trends")
            
        if not tab_list:
            st.info("This dataset does not contain enough variables to construct standard visualizations.")
        else:
            tabs = st.tabs(tab_list)
            
            # Tab 1: Numerical relationships and heatmaps
            if "🔗 Variable Relationships" in tab_list:
                with tabs[tab_list.index("🔗 Variable Relationships")]:
                    st.subheader("Correlation Heatmap & Custom Scatter Analytics")
                    
                    num_cols = col_types["numerical"]
                    
                    if len(num_cols) >= 2:
                        corr_matrix = df_clean[num_cols].corr().round(2)
                        fig_heat = px.imshow(
                            corr_matrix.values,
                            x=num_cols,
                            y=num_cols,
                            text_auto=True,
                            title="Correlation Heatmap (Cleaned Numerical Columns)",
                            color_continuous_scale="RdBu_r"
                        )
                        fig_heat.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
                        st.plotly_chart(fig_heat, width="stretch")
                    else:
                        st.write("Need at least 2 numerical columns to render a correlation heatmap.")
                        
                    # Custom Scatter Plot Builder
                    st.markdown("#### Custom Scatter Plot Constructor")
                    col_sc1, col_sc2, col_sc3 = st.columns(3)
                    with col_sc1:
                        sc_x = st.selectbox("Select X-Axis Variable", num_cols, key="sc_x")
                    with col_sc2:
                        sc_y = st.selectbox("Select Y-Axis Variable", num_cols, key="sc_y")
                    with col_sc3:
                        all_cols = col_types["numerical"] + col_types["categorical"]
                        sc_color = st.selectbox("Color Code By (Optional)", [None] + all_cols, key="sc_color")
                        
                    fig_scatter = px.scatter(
                        df_clean,
                        x=sc_x,
                        y=sc_y,
                        color=sc_color if sc_color else None,
                        title=f"Scatter Analysis: {sc_x} vs {sc_y}",
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    fig_scatter.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
                    st.plotly_chart(fig_scatter, width="stretch")

            # Tab 2: Categorical + Numerical Aggregations
            if "📊 Segment Aggregations" in tab_list:
                with tabs[tab_list.index("📊 Segment Aggregations")]:
                    st.subheader("Categorical Segment Aggregations")
                    
                    col_ag1, col_ag2, col_ag3 = st.columns(3)
                    with col_ag1:
                        group_col = st.selectbox("Group By Column", col_types["categorical"], key="grp_col")
                    with col_ag2:
                        value_col = st.selectbox("Numerical Aggregate Column", col_types["numerical"], key="val_col")
                    with col_ag3:
                        agg_func = st.selectbox("Aggregation Function", ["Sum", "Mean", "Min", "Max", "Count"], key="agg_fn")
                        
                    # Run aggregation
                    agg_mapping = {
                        "Sum": "sum",
                        "Mean": "mean",
                        "Min": "min",
                        "Max": "max",
                        "Count": "count"
                    }
                    
                    df_agg = df_clean.groupby(group_col)[value_col].agg(agg_mapping[agg_func]).reset_index()
                    df_agg.columns = [group_col, f"{agg_func} of {value_col}"]
                    
                    # Sort aggregated results
                    df_agg = df_agg.sort_values(by=f"{agg_func} of {value_col}", ascending=False).head(20)
                    
                    col_ch1, col_ch2 = st.columns(2)
                    with col_ch1:
                        fig_agg_bar = px.bar(
                            df_agg,
                            x=group_col,
                            y=f"{agg_func} of {value_col}",
                            title=f"Top 20 {group_col} by {agg_func} of {value_col}",
                            color=group_col,
                            color_discrete_sequence=px.colors.qualitative.Prism
                        )
                        fig_agg_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC", showlegend=False)
                        st.plotly_chart(fig_agg_bar, width="stretch")
                        
                    with col_ch2:
                        fig_agg_pie = px.pie(
                            df_agg,
                            names=group_col,
                            values=f"{agg_func} of {value_col}",
                            title=f"Relative Share by {group_col}",
                            color_discrete_sequence=px.colors.qualitative.Prism
                        )
                        fig_agg_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
                        st.plotly_chart(fig_agg_pie, width="stretch")

            # Tab 3: Chronological Trends
            if "📈 Chronological Trends" in tab_list:
                with tabs[tab_list.index("📈 Chronological Trends")]:
                    st.subheader("Chronological Trends (Time Series)")
                    
                    col_t1, col_t2 = st.columns(2)
                    with col_t1:
                        date_col = st.selectbox("Date Column", col_types["datetime"], key="dt_col")
                    with col_t2:
                        y_trend_col = st.selectbox("Numerical Trend Column", col_types["numerical"], key="y_trend")
                        
                    # Pre-process dates to ensure they parse for plotting
                    df_clean["Temp_Date_Dt"] = pd.to_datetime(df_clean[date_col])
                    daily_trends = df_clean.groupby("Temp_Date_Dt")[y_trend_col].sum().reset_index()
                    
                    fig_trend = px.line(
                        daily_trends,
                        x="Temp_Date_Dt",
                        y=y_trend_col,
                        title=f"Daily Sum of {y_trend_col} Over Time",
                        color_discrete_sequence=["#10B981"]
                    )
                    fig_trend.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
                    st.plotly_chart(fig_trend, width="stretch")
