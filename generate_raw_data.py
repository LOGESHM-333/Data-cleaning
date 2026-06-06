import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_messy_data(num_rows=1000):
    np.random.seed(42)
    random.seed(42)

    # Base transaction IDs
    tx_ids = [f"TX{10000 + i}" for i in range(num_rows - 50)] # 50 rows will be duplicate IDs
    # Add duplicate IDs to simulate transactional duplications
    tx_ids = tx_ids + random.choices(tx_ids[:100], k=50)
    random.shuffle(tx_ids)

    # Customer IDs
    customer_ids = [f"CUST{random.randint(1000, 2000)}" for _ in range(num_rows)]
    # Introduce some missing customer IDs
    for i in range(int(num_rows * 0.05)):
        customer_ids[random.randint(0, num_rows - 1)] = np.nan

    # Dates
    start_date = datetime(2025, 1, 1)
    dates = []
    date_formats = [
        "%Y-%m-%d",      # 2025-05-12
        "%d/%m/%Y",      # 12/05/2025
        "%m-%d-%Y",      # 05-12-2025
        "%B %d, %Y",     # May 12, 2025
    ]
    for _ in range(num_rows):
        days_offset = random.randint(0, 450)
        dt = start_date + timedelta(days=days_offset)
        
        # Inconsistently format date strings
        fmt = random.choice(date_formats)
        dates.append(dt.strftime(fmt))
    
    # Introduce some missing, corrupted, or future dates
    for _ in range(int(num_rows * 0.03)):
        dates[random.randint(0, num_rows - 1)] = np.nan
    for _ in range(int(num_rows * 0.02)):
        dates[random.randint(0, num_rows - 1)] = "2029-12-31" # Future date outlier
    for _ in range(int(num_rows * 0.01)):
        dates[random.randint(0, num_rows - 1)] = "N/A" # Corrupted date string

    # Ages
    ages = []
    for _ in range(num_rows):
        age_type = random.choices(["normal", "outlier_high", "outlier_negative"], weights=[0.92, 0.05, 0.03])[0]
        if age_type == "normal":
            ages.append(random.randint(18, 70))
        elif age_type == "outlier_high":
            ages.append(random.randint(120, 200)) # Impossibly high
        else:
            ages.append(random.randint(-15, -1)) # Negative ages

    # Add missing ages
    ages = [age if random.random() > 0.06 else np.nan for age in ages]

    # Gender with spelling variants and missing values
    genders = []
    gender_options = ["Male", "Female", "M", "F", "male", "female", "FEMALE", "MALE", "Unknown"]
    for _ in range(num_rows):
        if random.random() < 0.05:
            genders.append(np.nan)
        else:
            genders.append(random.choice(gender_options))

    # Product Category with inconsistencies
    categories = []
    cat_options = [
        "Electronics", "electronics", "ELEC", "Electronics & Accessories",
        "Clothing", "clothing", "CLOTHES", "Apparel",
        "Home & Kitchen", "home", "Home", "KITCHEN",
        "Books", "books", "BOOKS", "Reading Material"
    ]
    for _ in range(num_rows):
        if random.random() < 0.04:
            categories.append(np.nan)
        else:
            categories.append(random.choice(cat_options))

    # Quantities (should be positive integers, but we introduce negative values, zeroes, missing, and outliers)
    quantities = []
    for _ in range(num_rows):
        q_type = random.choices(["normal", "zero", "negative", "high_outlier"], weights=[0.90, 0.04, 0.04, 0.02])[0]
        if q_type == "normal":
            quantities.append(random.randint(1, 10))
        elif q_type == "zero":
            quantities.append(0)
        elif q_type == "negative":
            quantities.append(random.randint(-5, -1))
        else:
            quantities.append(random.randint(150, 500)) # Bulk order outlier
            
    # Add missing quantities
    quantities = [q if random.random() > 0.05 else np.nan for q in quantities]

    # Unit Prices
    unit_prices = []
    for _ in range(num_rows):
        p_type = random.choices(["normal", "negative", "huge_outlier"], weights=[0.94, 0.03, 0.03])[0]
        if p_type == "normal":
            unit_prices.append(round(random.uniform(10.0, 500.0), 2))
        elif p_type == "negative":
            unit_prices.append(round(random.uniform(-50.0, -5.0), 2))
        else:
            unit_prices.append(round(random.uniform(10000.0, 99999.0), 2)) # Impossibly high prices

    # Add missing unit prices
    unit_prices = [p if random.random() > 0.05 else np.nan for p in unit_prices]

    # Total Amount: should be Quantity * Unit Price, but we will introduce errors and nulls
    total_amounts = []
    for i in range(num_rows):
        q = quantities[i]
        p = unit_prices[i]
        
        if pd.isna(q) or pd.isna(p):
            total_amounts.append(np.nan)
            continue
            
        # Introduce mismatch/errors (10% of the time, write incorrect total amount)
        if random.random() < 0.10:
            total_amounts.append(round(q * p * random.uniform(0.5, 1.5), 2))
        elif random.random() < 0.05:
            total_amounts.append(np.nan)
        else:
            total_amounts.append(round(q * p, 2))

    # Payment Methods
    payment_methods = []
    pay_options = ["Credit Card", "credit card", "Debit Card", "PayPal", "paypal", "Cash", "CASH", "E-Wallet"]
    for _ in range(num_rows):
        if random.random() < 0.05:
            payment_methods.append(np.nan)
        else:
            payment_methods.append(random.choice(pay_options))

    # Customer Ratings
    ratings = []
    for _ in range(num_rows):
        r_type = random.choices(["normal", "low_outlier", "high_outlier"], weights=[0.90, 0.05, 0.05])[0]
        if r_type == "normal":
            ratings.append(random.randint(1, 5))
        elif r_type == "low_outlier":
            ratings.append(random.randint(-5, 0)) # Invalid negative ratings
        else:
            ratings.append(random.randint(10, 20)) # Out of range high ratings

    # Add missing ratings
    ratings = [r if random.random() > 0.08 else np.nan for r in ratings]

    # Construct DataFrame
    df = pd.DataFrame({
        "Transaction_ID": tx_ids,
        "Date": dates,
        "Customer_ID": customer_ids,
        "Customer_Age": ages,
        "Customer_Gender": genders,
        "Product_Category": categories,
        "Quantity": quantities,
        "Unit_Price": unit_prices,
        "Total_Amount": total_amounts,
        "Payment_Method": payment_methods,
        "Customer_Rating": ratings
    })

    # Add 20 completely duplicate rows
    duplicate_rows = df.sample(n=20, random_state=42)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    return df

if __name__ == "__main__":
    print("Generating raw, messy e-commerce dataset...")
    df = generate_messy_data(num_rows=1200)
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save raw CSV
    output_path = "data/raw_sales_data.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Dataset saved to '{output_path}'.")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Print some stats of anomalies
    print("\nAnomaly Stats:")
    print(f"Missing values per column:\n{df.isnull().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Negative customer age count: {(df['Customer_Age'] < 0).sum()}")
    print(f"Outlier customer age count (>100): {(df['Customer_Age'] > 100).sum()}")
    print(f"Negative quantities count: {(df['Quantity'] < 0).sum()}")
    print(f"Extreme prices count (>5000): {(df['Unit_Price'] > 5000).sum()}")
