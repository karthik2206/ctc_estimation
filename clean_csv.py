import pandas as pd
import numpy as np

def clean_column_names(df):
    """Convert column names to lowercase, replace spaces with underscores"""
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def remove_columns(df):
    """Remove specified columns"""
    columns_to_remove = [
        'last_name', 'email_id', 'expected_ctc', 'notice_period', 'specialization', 'phone_number',
        # Add company type columns
        'company_type_1', 'company_type_2', 'company_type_3', 'company_type_4', 'company_type_5',
        # Add funding status columns
        'is_funded_1', 'is_funded_2', 'is_funded_3', 'is_funded_4', 'is_funded_5'
    ]
    return df.drop(columns=columns_to_remove, errors='ignore')

def clean_yoe(value):
    """Round down Years of Experience"""
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None

def clean_current_ctc(value):
    """Remove 'LPA' from Current CTC and convert to float"""
    try:
        if isinstance(value, str) and 'LPA' in value:
            return float(value.replace('LPA', '').strip())
        return float(value)
    except (ValueError, TypeError):
        return None

def map_tier(avg_value):
    """Map averaged values to college tiers"""
    rounded_value = round(avg_value)
    tier_mapping = {
        1: 'No Tier',
        2: 'Tier 3',
        3: 'Tier 2',
        4: 'Tier 1',
        5: 'Elite'
    }
    # Ensure the value is within valid range
    rounded_value = max(1, min(5, rounded_value))
    return tier_mapping[rounded_value]

def clean_college_type(value):
    """Calculate average of comma-separated numbers and map to tiers"""
    try:
        if isinstance(value, str):
            numbers = [float(num.strip()) for num in value.split(',')]
            avg = np.mean(numbers)
            return map_tier(avg)
        return map_tier(float(value))
    except (ValueError, TypeError):
        return 'No Tier'

def determine_company_category(row):
    """
    Determine company category based on company type and funding status
    Returns one of: Product-funded, Product-unfunded, Services-funded, Services-unfunded, Not-valid
    """
    # Check if all relevant columns are empty
    company_type_cols = [f'company_type_{i}' for i in range(1, 6)]
    is_funded_cols = [f'is_funded_{i}' for i in range(1, 6)]
    
    # If all columns are empty or NaN, return Not-valid
    if row[company_type_cols + is_funded_cols].isna().all():
        return 'Not-valid'
    
    # Determine if it's a Product company
    is_product = any(
        str(row[col]).lower() == 'product company' 
        for col in company_type_cols 
        if pd.notna(row[col])
    )
    
    # Determine if it's funded
    is_funded = any(
        str(row[col]).lower() == 'yes'
        for col in is_funded_cols
        if pd.notna(row[col])
    )
    
    # Create category string
    company_type = 'Product' if is_product else 'Services'
    funding_status = 'funded' if is_funded else 'unfunded'
    
    return f'{company_type}-{funding_status}'

def create_candidate_category(row):
    """Create candidate category based on YOE, College Tier, and Company Category"""
    yoe = row['yoe']
    college = row['college_type']
    company = row['company_category']
    
    # Define YOE ranges
    if yoe < 1:
        yoe_category = "0-1"
    elif yoe >= 1 and yoe < 3:
        yoe_category = "1-3"
    elif yoe >= 3 and yoe < 5:
        yoe_category = "3-5"
    elif yoe >= 5 and yoe < 7:
        yoe_category = "5-7"
    elif yoe >= 7 and yoe < 9:
        yoe_category = "7-9"
    elif yoe >= 9 and yoe < 11:
        yoe_category = "9-11"
    elif yoe >= 11 and yoe < 13:
        yoe_category = "11-13"
    elif yoe >= 13 and yoe < 15:
        yoe_category = "13-15"
    else:
        yoe_category = "15+"
    
    return f"{yoe_category}yrs-{college}-{company}"

def remove_ctc_outliers_by_category(df):
    """Remove top and bottom 2% CTC outliers within each candidate category"""
    initial_total = len(df)
    
    # Create a copy to avoid SettingWithCopyWarning
    df_cleaned = df.copy()
    
    # Group by candidate category and remove outliers within each group
    categories = df_cleaned['candidate_category'].unique()
    rows_to_keep = []
    
    print("\nRemoving CTC outliers within each category:")
    for category in categories:
        category_mask = df_cleaned['candidate_category'] == category
        category_df = df_cleaned[category_mask]
        
        if len(category_df) < 50:  # Skip small categories to avoid over-filtering
            rows_to_keep.append(category_df)
            continue
            
        lower_bound = category_df['current_ctc'].quantile(0.10)
        upper_bound = category_df['current_ctc'].quantile(0.90)
        
        filtered_df = category_df[
            (category_df['current_ctc'] >= lower_bound) & 
            (category_df['current_ctc'] <= upper_bound)
        ]
        
        removed_count = len(category_df) - len(filtered_df)
        if removed_count > 0:
            print(f"  {category}: removed {removed_count} rows (CTC range: {lower_bound:.2f} to {upper_bound:.2f} LPA)")
        
        rows_to_keep.append(filtered_df)
    
    # Combine all filtered data
    df_cleaned = pd.concat(rows_to_keep)
    total_removed = initial_total - len(df_cleaned)
    print(f"\nTotal rows removed as outliers: {total_removed}")
    
    return df_cleaned

def clean_csv(input_file, output_file):
    """Main function to clean the CSV file"""
    print(f"\nStarting CSV cleaning process...")
    print(f"Reading input file: {input_file}")
    # Read the CSV file
    df = pd.read_csv(input_file)
    print(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    
    print("\nCleaning column names...")
    # Clean column names
    df = clean_column_names(df)
    print("Column names cleaned and standardized")
    
    print("\nCalculating company categories...")
    # Add company category column (do this before removing columns)
    df['company_category'] = df.apply(determine_company_category, axis=1)
    category_counts = df['company_category'].value_counts()
    print("Company categories calculated. Distribution:")
    print(category_counts)
    
    print("\nRemoving unnecessary columns...")
    # Remove specified columns (now includes company type and funding columns)
    initial_cols = set(df.columns)
    df = remove_columns(df)
    removed_cols = initial_cols - set(df.columns)
    print(f"Removed {len(removed_cols)} columns: {', '.join(removed_cols)}")
    
    print("\nCleaning numeric and categorical columns...")
    # Clean YOE column
    if 'yoe' in df.columns:
        df['yoe'] = df['yoe'].apply(clean_yoe)
        print("Years of Experience cleaned")
    
    # Clean Current CTC column
    if 'current_ctc' in df.columns:
        df['current_ctc'] = df['current_ctc'].apply(clean_current_ctc)
        print("Current CTC cleaned")
        
        # Remove rows where current_ctc is 0
        initial_rows = len(df)
        df = df[df['current_ctc'] != 0]
        removed_rows = initial_rows - len(df)
        print(f"Removed {removed_rows} rows where Current CTC was 0")
        
        # Remove top and bottom 5% outliers from overall data
        initial_rows = len(df)
        lower_bound = df['current_ctc'].quantile(0.05)
        upper_bound = df['current_ctc'].quantile(0.95)
        df = df[(df['current_ctc'] >= lower_bound) & (df['current_ctc'] <= upper_bound)]
        removed_rows = initial_rows - len(df)
        print(f"Removed {removed_rows} rows as overall outliers (top and bottom 5%)")
        print(f"Overall CTC range after outlier removal: {lower_bound:.2f} to {upper_bound:.2f} LPA")
    
    # Clean College Type column
    if 'college_type' in df.columns:
        df['college_type'] = df['college_type'].apply(clean_college_type)
        print("College Type cleaned")
    
    print("\nCreating candidate categories...")
    # Create candidate categories
    df['candidate_category'] = df.apply(create_candidate_category, axis=1)
    category_counts = df['candidate_category'].value_counts()
    print("\nYears of Experience distribution:")
    print(df['yoe'].describe())
    print("\nTop 10 most common candidate categories:")
    print(category_counts.head(10))
    
    # Remove outliers within each category
    df = remove_ctc_outliers_by_category(df)
    
    print(f"\nSaving cleaned CSV to: {output_file}")
    # Save the cleaned CSV
    df.to_csv(output_file, index=False)
    print(f"Cleaning complete! Final CSV has {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns in final CSV: {', '.join(df.columns)}")

if __name__ == "__main__":
    input_file = "/Users/karthiksridharan/Desktop/talent_profiles_export.csv"  # Replace with your input file path
    output_file = "/Users/karthiksridharan/Desktop/talent_profiles_export_output_v4.csv"  # Replace with your desired output file path
    clean_csv(input_file, output_file) 