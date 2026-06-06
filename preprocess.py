import os
import re
import pandas as pd
import numpy as np

def detect_column_types(df):
    """
    Automatically infers column types into:
    - numerical: integers and floats
    - datetime: columns that parse successfully as dates
    - categorical: object/string/boolean columns
    """
    column_types = {
        "numerical": [],
        "datetime": [],
        "categorical": []
    }
    
    for col in df.columns:
        # 1. Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # Sometimes low-cardinality integers are categorical, but keeping them as numerical is safer
            column_types["numerical"].append(col)
            continue
            
        # 2. Check if column is datetime or can be parsed as datetime
        # If the column has datetime objects already
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types["datetime"].append(col)
            continue
            
        # If the column is string/object, try parsing a sample of non-null values
        non_null_samples = df[col].dropna().head(100)
        if len(non_null_samples) > 0:
            parsed_count = 0
            for val in non_null_samples:
                # Basic string length or regex check to avoid parsing single numbers as dates
                val_str = str(val).strip()
                if len(val_str) < 5 or val_str.isdigit():
                    continue
                try:
                    pd.to_datetime(val_str, errors='raise')
                    parsed_count += 1
                except (ValueError, TypeError, OverflowError):
                    pass
            
            # If over 60% of samples are parseable as date, classify as datetime
            if (parsed_count / len(non_null_samples)) > 0.6:
                column_types["datetime"].append(col)
                continue
                
        # 3. Otherwise, classify as categorical
        column_types["categorical"].append(col)
        
    return column_types

def clean_generic_dataframe(df_raw, config=None):
    """
    Cleans an arbitrary dataframe based on the provided configuration dictionary.
    Returns: (cleaned_df, column_types, change_log)
    """
    df = df_raw.copy()
    
    if config is None:
        config = {
            "drop_duplicates": True,
            "numerical": {
                "impute_strategy": "median",      # median, mean, mode, zero, none
                "clamp_outliers": True,
                "outlier_iqr_k": 1.5
            },
            "categorical": {
                "impute_strategy": "placeholder", # mode, placeholder, none
                "placeholder_value": "Unknown",
                "normalize_case": "Title Case",   # Title Case, lowercase, UPPERCASE, none
                "strip_whitespace": True
            },
            "datetime": {
                "impute_strategy": "median",      # median, now, none
            }
        }
        
    change_log = {}
    
    # 1. Handle Duplicates
    if config.get("drop_duplicates", True):
        initial_len = len(df)
        df = df.drop_duplicates()
        removed = initial_len - len(df)
        if removed > 0:
            change_log["duplicates_removed"] = removed
            
    # 2. Detect Columns
    col_types = detect_column_types(df)
    
    # Process Numerical Columns
    num_config = config.get("numerical", {})
    for col in col_types["numerical"]:
        col_log = {}
        
        # Missing values detection
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            strategy = num_config.get("impute_strategy", "median")
            if strategy == "median":
                fill_val = df[col].median()
                df[col] = df[col].fillna(fill_val)
                col_log["imputed_missing"] = f"Filled {missing_count} nulls using Median ({fill_val})"
            elif strategy == "mean":
                fill_val = df[col].mean()
                df[col] = df[col].fillna(fill_val)
                col_log["imputed_missing"] = f"Filled {missing_count} nulls using Mean ({round(fill_val, 2)})"
            elif strategy == "mode":
                fill_val = df[col].mode().iloc[0] if not df[col].mode().empty else 0
                df[col] = df[col].fillna(fill_val)
                col_log["imputed_missing"] = f"Filled {missing_count} nulls using Mode ({fill_val})"
            elif strategy == "zero":
                df[col] = df[col].fillna(0)
                col_log["imputed_missing"] = f"Filled {missing_count} nulls with 0"
                
        # Outlier Detection & Clamping
        if num_config.get("clamp_outliers", True) and len(df[col].dropna()) > 0:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            k = num_config.get("outlier_iqr_k", 1.5)
            
            lower_bound = q1 - k * iqr
            upper_bound = q3 + k * iqr
            
            outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers_count = outliers_mask.sum()
            
            if outliers_count > 0:
                # Clamp values
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                col_log["outliers_clamped"] = f"Clamped {outliers_count} outliers to range [{round(lower_bound, 2)}, {round(upper_bound, 2)}]"
                
        if col_log:
            change_log[col] = col_log

    # Process Datetime Columns
    dt_config = config.get("datetime", {})
    for col in col_types["datetime"]:
        col_log = {}
        
        # Standardise date formatting
        initial_nulls = df[col].isnull().sum()
        df[col] = pd.to_datetime(df[col], errors='coerce')
        failed_parses = df[col].isnull().sum() - initial_nulls
        
        if failed_parses > 0:
            col_log["parsing_failures"] = f"Failed to parse {failed_parses} records into dates (set to null)"
            
        missing_dates = df[col].isnull().sum()
        if missing_dates > 0:
            strategy = dt_config.get("impute_strategy", "median")
            if strategy == "median":
                valid_dates = df[col].dropna()
                if len(valid_dates) > 0:
                    fill_date = valid_dates.sort_values().iloc[len(valid_dates) // 2]
                else:
                    fill_date = pd.Timestamp.now()
                df[col] = df[col].fillna(fill_date)
                col_log["imputed_missing"] = f"Filled {missing_dates} missing dates with Median ({fill_date.strftime('%Y-%m-%d')})"
            elif strategy == "now":
                fill_date = pd.Timestamp.now()
                df[col] = df[col].fillna(fill_date)
                col_log["imputed_missing"] = f"Filled {missing_dates} missing dates with current time ({fill_date.strftime('%Y-%m-%d')})"
                
        # Format as YYYY-MM-DD
        df[col] = df[col].dt.date
        
        if col_log:
            change_log[col] = col_log

    # Process Categorical Columns
    cat_config = config.get("categorical", {})
    for col in col_types["categorical"]:
        col_log = {}
        
        # Convert to string and handle nulls
        missing_count = df[col].isnull().sum()
        
        # Whitespace
        if cat_config.get("strip_whitespace", True):
            df[col] = df[col].apply(lambda x: str(x).strip() if pd.notna(x) else x)
            
        # Case conversion
        casing = cat_config.get("normalize_case", "Title Case")
        if casing == "Title Case":
            df[col] = df[col].apply(lambda x: str(x).title() if pd.notna(x) else x)
        elif casing == "lowercase":
            df[col] = df[col].apply(lambda x: str(x).lower() if pd.notna(x) else x)
        elif casing == "UPPERCASE":
            df[col] = df[col].apply(lambda x: str(x).upper() if pd.notna(x) else x)
            
        # Imputation
        if missing_count > 0:
            strategy = cat_config.get("impute_strategy", "placeholder")
            if strategy == "placeholder":
                val = cat_config.get("placeholder_value", "Unknown")
                # Handle title/lower/upper casing of placeholder
                if casing == "Title Case": val = val.title()
                elif casing == "lowercase": val = val.lower()
                elif casing == "UPPERCASE": val = val.upper()
                
                df[col] = df[col].fillna(val)
                col_log["imputed_missing"] = f"Filled {missing_count} missing values with placeholder '{val}'"
            elif strategy == "mode":
                modes = df[col].dropna().mode()
                val = modes.iloc[0] if not modes.empty else "Unknown"
                df[col] = df[col].fillna(val)
                col_log["imputed_missing"] = f"Filled {missing_count} missing values with Mode ('{val}')"
                
        if col_log:
            change_log[col] = col_log

    return df, col_types, change_log


# Keep the original function for the pre-generated messy dataset demo!
def clean_dataframe(df_raw):
    """
    Specialized cleaner for the generated messy_sales_data.csv.
    """
    df = df_raw.copy()
    stats = {}

    # 1. Handle Duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    stats["duplicates_removed"] = initial_rows - len(df)

    # 2. Standardise Date Column
    parsed_dates = []
    for d in df["Date"]:
        if pd.isna(d) or str(d).strip().lower() in ["n/a", "unknown", "nan", "null"]:
            parsed_dates.append(pd.NaT)
            continue
        
        parsed = None
        for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%B %d, %Y"]:
            try:
                parsed = pd.to_datetime(str(d).strip(), format=fmt)
                break
            except (ValueError, TypeError):
                continue
        
        if parsed is None:
            try:
                parsed = pd.to_datetime(str(d).strip(), errors='coerce')
            except:
                parsed = pd.NaT
        parsed_dates.append(parsed)

    df["Date"] = parsed_dates
    missing_dates = df["Date"].isna().sum()
    stats["missing_dates_found"] = missing_dates

    valid_dates = df["Date"].dropna()
    median_date = valid_dates.sort_values().iloc[len(valid_dates) // 2] if len(valid_dates) > 0 else pd.Timestamp.now()
    df["Date"] = df["Date"].fillna(median_date)

    future_mask = df["Date"] > pd.Timestamp.now()
    stats["future_dates_corrected"] = future_mask.sum()
    df.loc[future_mask, "Date"] = median_date
    df["Date"] = df["Date"].dt.date

    # 3. Standardise Gender
    gender_map = {"m": "Male", "male": "Male", "f": "Female", "female": "Female", "unknown": "Unknown"}
    df["Customer_Gender"] = df["Customer_Gender"].apply(lambda val: gender_map.get(str(val).strip().lower(), "Unknown") if pd.notna(val) else "Unknown")
    stats["missing_genders_imputed"] = (df["Customer_Gender"] == "Unknown").sum()

    # 4. Standardise Product Category
    category_patterns = [
        (r"elect|elec|accessories", "Electronics"),
        (r"cloth|apparel", "Clothing"),
        (r"home|kitchen|appliances", "Home & Kitchen"),
        (r"book|read", "Books")
    ]
    def clean_category(val):
        if pd.isna(val) or str(val).strip().lower() in ["nan", "null", "n/a", ""]: return "Unknown"
        for pattern, replacement in category_patterns:
            if re.search(pattern, str(val).strip(), re.IGNORECASE): return replacement
        return "Unknown"
    df["Product_Category"] = df["Product_Category"].apply(clean_category)
    stats["missing_categories_imputed"] = (df["Product_Category"] == "Unknown").sum()

    # 5. Customer Age Cleaning
    invalid_age_mask = (df["Customer_Age"] < 18) | (df["Customer_Age"] > 100) | df["Customer_Age"].isna()
    stats["missing_ages_imputed"] = df["Customer_Age"].isna().sum()
    valid_ages = df.loc[~invalid_age_mask, "Customer_Age"]
    median_age = int(valid_ages.median()) if len(valid_ages) > 0 else 35
    df["Customer_Age"] = df["Customer_Age"].fillna(median_age)
    df.loc[(df["Customer_Age"] < 18) | (df["Customer_Age"] > 100), "Customer_Age"] = median_age
    df["Customer_Age"] = df["Customer_Age"].astype(int)
    stats["invalid_ages_corrected"] = invalid_age_mask.sum()

    # 6. Quantity Cleaning
    invalid_qty_mask = (df["Quantity"] <= 0) | df["Quantity"].isna()
    stats["missing_quantities_imputed"] = df["Quantity"].isna().sum()
    valid_qtys = df.loc[~invalid_qty_mask, "Quantity"]
    median_qty = int(valid_qtys.median()) if len(valid_qtys) > 0 else 2
    df["Quantity"] = df["Quantity"].fillna(median_qty)
    df.loc[df["Quantity"] <= 0, "Quantity"] = median_qty
    
    q_q1 = df["Quantity"].quantile(0.25)
    q_q3 = df["Quantity"].quantile(0.75)
    q_upper = q_q3 + 1.5 * (q_q3 - q_q1)
    qty_outlier_mask = df["Quantity"] > q_upper
    stats["quantity_outliers_clamped"] = qty_outlier_mask.sum()
    df.loc[qty_outlier_mask, "Quantity"] = int(q_upper)
    df["Quantity"] = df["Quantity"].astype(int)

    # 7. Unit Price Cleaning
    invalid_price_mask = (df["Unit_Price"] <= 0) | df["Unit_Price"].isna()
    stats["invalid_prices_corrected"] = invalid_price_mask.sum()
    
    # Category based median
    median_prices_by_cat = {}
    for cat in df["Product_Category"].unique():
        cat_mask = (df["Product_Category"] == cat) & (df["Unit_Price"] > 0) & (~df["Unit_Price"].isna())
        cat_prices = df.loc[cat_mask, "Unit_Price"]
        if len(cat_prices) > 0:
            median_prices_by_cat[cat] = cat_prices[cat_prices <= cat_prices.quantile(0.95)].median()
        else:
            median_prices_by_cat[cat] = 49.99
            
    for idx, row in df.iterrows():
        if pd.isna(row["Unit_Price"]) or row["Unit_Price"] <= 0:
            df.at[idx, "Unit_Price"] = median_prices_by_cat[row["Product_Category"]]

    price_outliers_clamped = 0
    for cat in df["Product_Category"].unique():
        cat_mask = df["Product_Category"] == cat
        cat_df = df.loc[cat_mask]
        p1 = cat_df["Unit_Price"].quantile(0.25)
        p3 = cat_df["Unit_Price"].quantile(0.75)
        upper_limit = p3 + 1.5 * (p3 - p1)
        cat_outliers = (df["Product_Category"] == cat) & (df["Unit_Price"] > upper_limit)
        price_outliers_clamped += cat_outliers.sum()
        df.loc[cat_outliers, "Unit_Price"] = round(upper_limit, 2)
    stats["price_outliers_clamped"] = price_outliers_clamped
    df["Unit_Price"] = df["Unit_Price"].round(2)

    # 8. Total Amount
    mismatch_mask = (df["Total_Amount"] != (df["Quantity"] * df["Unit_Price"]).round(2)) | df["Total_Amount"].isna()
    stats["total_amount_mismatches_fixed"] = mismatch_mask.sum()
    df["Total_Amount"] = (df["Quantity"] * df["Unit_Price"]).round(2)

    # 9. Payment Method
    missing_payments = df["Payment_Method"].isna().sum()
    stats["missing_payments_imputed"] = missing_payments
    def clean_payment(val):
        if pd.isna(val): return "Unknown"
        v_clean = str(val).strip().lower()
        if "paypal" in v_clean: return "PayPal"
        if "credit" in v_clean: return "Credit Card"
        if "debit" in v_clean: return "Debit Card"
        if "cash" in v_clean: return "Cash"
        if "wallet" in v_clean: return "E-Wallet"
        return v_clean.title()
    df["Payment_Method"] = df["Payment_Method"].apply(clean_payment)

    # 10. Customer Ratings
    invalid_rating_mask = (df["Customer_Rating"] < 1) | (df["Customer_Rating"] > 5) | df["Customer_Rating"].isna()
    stats["invalid_ratings_corrected"] = invalid_rating_mask.sum()
    valid_ratings = df.loc[~invalid_rating_mask, "Customer_Rating"]
    median_rating = round(valid_ratings.median()) if len(valid_ratings) > 0 else 4
    df["Customer_Rating"] = df["Customer_Rating"].fillna(median_rating)
    df.loc[(df["Customer_Rating"] < 1) | (df["Customer_Rating"] > 5), "Customer_Rating"] = median_rating
    df["Customer_Rating"] = df["Customer_Rating"].astype(int)

    df["Customer_ID"] = df["Customer_ID"].fillna("CUST_UNKNOWN")

    return df, stats
