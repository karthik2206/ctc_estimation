import pandas as pd
import numpy as np

def clean_column_names(df):
    """Convert column names to lowercase, replace spaces with underscores"""
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def categorize_city(city):
    """Categorize cities into Metro, Tier 1, Tier 2, and Other"""
    metro_cities = {'Bangalore', 'Gurgaon', 'Mumbai'}
    tier_1_cities = {'Pune', 'Noida', 'New Delhi', 'Ahmedabad', 'Hyderabad', 'Chennai'}
    tier_2_cities = {'Lucknow', 'Bhopal', 'Indore', 'Surat', 'Jaipur', 
                    'Chandigarh', 'Nagpur', 'Coimbatore'}
    
    if pd.isna(city):
        return 'Unknown'
    city = str(city).strip().title()
    
    if city in metro_cities:
        return 'Metro'
    if city in tier_1_cities:
        return 'Tier 1'
    if city in tier_2_cities:
        return 'Tier 2'
    return 'Other'

def get_company_type(row):
    """Get company type from latest entry (company_type_1)"""
    if pd.isna(row['company_type_1']):
        return 'Unknown'
    return 'Product' if row['company_type_1'] == 1 else 'Services'

def get_funding_status(row):
    """Get funding status from latest entry (is_funded_1)"""
    if pd.isna(row['is_funded_1']):
        return 'Unknown'
    return 'Funded' if row['is_funded_1'] == 1 else 'Unfunded'

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

def create_candidate_category(row):
    """Create candidate category based on YOE, College Tier, and Company Type/Funding"""
    yoe = row['yoe']
    college = row['college_type']
    company_type = row['company_type']
    funding = row['funding_status']
    
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
    
    return f"{yoe_category}yrs-{college}-{company_type}-{funding}"

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
            
        lower_bound = category_df['current_ctc'].quantile(0.02)
        upper_bound = category_df['current_ctc'].quantile(0.98)
        
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

def filter_ctc_by_experience(df):
    """
    Filter rows based on CTC requirements for different years of experience:
    - Minimum CTC of 3 LPA for all rows
    - For YOE >= 2, minimum CTC should be YOE * 2
    - Maximum CTC of 999 LPA (remove unrealistic values)
    """
    initial_rows = len(df)
    
    # Create a copy to avoid SettingWithCopyWarning
    df_cleaned = df.copy()
    
    # Remove rows with CTC < 3 or > 999
    df_cleaned = df_cleaned[
        (df_cleaned['current_ctc'] >= 3) & 
        (df_cleaned['current_ctc'] <= 999)
    ]
    
    # For YOE >= 2, apply the YOE * 2 rule
    mask = df_cleaned['yoe'] >= 2
    df_cleaned = df_cleaned[
        ~mask | (mask & (df_cleaned['current_ctc'] >= df_cleaned['yoe'] * 2))
    ]
    
    removed_rows = initial_rows - len(df_cleaned)
    print(f"\nFiltering based on CTC requirements:")
    print(f"  Removed {removed_rows} rows that didn't meet CTC requirements")
    print(f"  - All rows must have CTC >= 3 LPA and <= 999 LPA")
    print(f"  - For YOE >= 2, CTC must be >= YOE * 2 LPA")
    
    return df_cleaned

def remove_outliers_and_invalid_records(df):
    """Remove outliers and invalid records from the dataset"""
    print("\nRemoving outliers and invalid records...")
    initial_rows = len(df)
    
    # Remove records with missing values
    df = df.dropna(subset=['yoe', 'current_ctc', 'college_type', 'company_type', 'funding_status', 'city_category'])
    removed_missing = initial_rows - len(df)
    print(f"Removed {removed_missing} records with missing values")
    
    # Remove invalid YOE values (negative and unrealistic)
    df = df[df['yoe'] >= 0]  # Remove negative YOE
    df = df[df['yoe'] <= 40]  # Remove unrealistic YOE (>40 years)
    removed_yoe = initial_rows - removed_missing - len(df)
    print(f"Removed {removed_yoe} records with invalid YOE values")
    
    # Calculate salary statistics for outlier detection
    salary_mean = df['current_ctc'].mean()
    salary_std = df['current_ctc'].std()
    salary_p95 = df['current_ctc'].quantile(0.95)
    
    # Create a mask for problematic records
    problematic_mask = pd.Series(False, index=df.index)
    
    # 1. Check for minimum expected salary based on YOE
    expected_min_salary = df['yoe'].apply(lambda x: max(3.0, x * 2))
    problematic_mask |= (df['current_ctc'] < expected_min_salary)
    
    # 2. Check for high salary outliers
    problematic_mask |= (df['current_ctc'] > salary_p95)
    problematic_mask |= (abs(df['current_ctc'] - salary_mean) > 2.5 * salary_std)
    
    # 3. Basic salary range check
    problematic_mask |= (df['current_ctc'] < 3)  # Minimum 3 LPA
    problematic_mask |= (df['current_ctc'] > 99)  # Maximum 99 LPA
    
    # Remove problematic records
    df_clean = df[~problematic_mask]
    removed_problematic = len(df) - len(df_clean)
    print(f"Removed {removed_problematic} problematic records")
    
    # Print detailed statistics of removed records
    print("\nDetailed analysis of removed records:")
    df_removed = df[problematic_mask]
    
    print("\nSalary distribution of removed records:")
    print(df_removed['current_ctc'].describe())
    
    print("\nYOE distribution of removed records:")
    print(df_removed['yoe'].describe())
    
    print("\nCollege Type distribution of removed records:")
    print(df_removed['college_type'].value_counts())
    
    print("\nCompany Type distribution of removed records:")
    print(df_removed['company_type'].value_counts())
    
    # Print statistics of clean records
    print("\nStatistics of clean records:")
    print("\nSalary distribution:")
    print(df_clean['current_ctc'].describe())
    
    print("\nYOE distribution:")
    print(df_clean['yoe'].describe())
    
    print(f"\nFinal dataset has {len(df_clean)} records ({(len(df_clean)/initial_rows)*100:.1f}% of original)")
    return df_clean

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
    
    # Clean numeric and categorical columns
    print("\nCleaning numeric and categorical columns...")
    df['yoe'] = df['yoe'].apply(clean_yoe)
    print("Years of Experience cleaned")
    
    df['current_ctc'] = df['current_ctc'].apply(clean_current_ctc)
    print("Current CTC cleaned")
    
    # Clean and categorize other columns
    df['college_type'] = df['college_type'].apply(clean_college_type)
    print("College Type cleaned")
    
    df['city_category'] = df['current_city'].apply(categorize_city)
    print("City categorized")
    
    # Get company type and funding status
    df['company_type'] = df.apply(get_company_type, axis=1)
    df['funding_status'] = df.apply(get_funding_status, axis=1)
    print("Company type and funding status determined")
    
    # Remove outliers and invalid records
    df = remove_outliers_and_invalid_records(df)
    
    # Keep only necessary columns
    columns_to_keep = [
        'yoe', 'current_ctc', 'college_type', 'specialization',
        'city_category', 'company_type', 'funding_status'
    ]
    df = df[columns_to_keep]
    
    # Print summary statistics
    print("\nSummary statistics after cleaning:")
    print("\nYears of Experience distribution:")
    print(df['yoe'].describe())
    print("\nCurrent CTC distribution:")
    print(df['current_ctc'].describe())
    print("\nCollege Type distribution:")
    print(df['college_type'].value_counts())
    print("\nCompany Type distribution:")
    print(df['company_type'].value_counts())
    print("\nFunding Status distribution:")
    print(df['funding_status'].value_counts())
    print("\nCity Category distribution:")
    print(df['city_category'].value_counts())
    
    # Save cleaned CSV
    df.to_csv(output_file, index=False)
    print(f"\nCleaning complete! Final CSV has {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns in final CSV: {', '.join(df.columns)}")
    
    return df

if __name__ == "__main__":
    input_file = "/Users/karthiksridharan/Desktop/talent_profiles_export.csv"
    output_file = "/Users/karthiksridharan/Desktop/talent_profiles_export_output_v4.csv"
    clean_csv(input_file, output_file) 