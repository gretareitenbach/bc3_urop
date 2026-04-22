import pandas as pd
import statsmodels.api as sm
import sys

def run_spatial_regression(input_file, output_file):
    """
    Runs a multilinear regression modeling the local climate change slope
    based on geographical and climatological factors.
    """
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print("Error: File not found. Please check the path.")
        return

    # 1. Define the Target (Dependent Variable) and Features (Independent Variables)
    # We assume 'slope' is the local climate change trend we want to explain.
    target_col = 'slope'

    feature_cols = [
        'climo_temp',
        'climo_humidity'
    ]

    # 2. Data Cleaning
    # Ensure all necessary columns exist
    missing_cols = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing_cols:
        print(f"Error: The following columns are missing from the CSV: {missing_cols}")
        return

    # Drop rows containing missing values in the relevant columns
    initial_count = len(df)
    df_clean = df.dropna(subset=[target_col] + feature_cols)
    cleaned_count = len(df_clean)

    if cleaned_count < initial_count:
        print(f"Dropped {initial_count - cleaned_count} rows due to missing data.")

    if cleaned_count == 0:
        print("Error: No data remaining after cleaning.")
        return

    # 3. Prepare Regression Data
    X = df_clean[feature_cols]
    y = df_clean[target_col]

    # Add a constant to the model (the intercept term)
    X = sm.add_constant(X)

    # 4. Run the OLS (Ordinary Least Squares) Regression
    print("Fitting OLS regression model...")
    model = sm.OLS(y, X).fit()

    # 5. Extract Results for CSV Output
    # We will create a DataFrame that holds the coefficients and stats
    results_data = []

    # Get the model-wide statistics
    r_squared = model.rsquared
    adj_r_squared = model.rsquared_adj
    f_pvalue = model.f_pvalue

    # Iterate through each predictor (including intercept/const)
    for term in model.params.index:
        row = {
            'Factor': term,
            'Coefficient (Slope)': model.params[term],
            'Std Error': model.bse[term],
            'P-Value': model.pvalues[term],
            'T-Value': model.tvalues[term],
            'Conf_Int_Low (2.5%)': model.conf_int().loc[term][0],
            'Conf_Int_High (97.5%)': model.conf_int().loc[term][1],
            # Add model-wide stats to every row (or just the first) for reference
            'Model_R_Squared': r_squared,
            'Model_Adj_R_Squared': adj_r_squared,
            'Model_Prob(F-Stat)': f_pvalue,
            'Observations': model.nobs
        }
        results_data.append(row)

    # 6. Save to CSV
    results_df = pd.DataFrame(results_data)

    # Reorder columns to put the most important ones first
    cols = ['Factor', 'Coefficient (Slope)', 'P-Value', 'Model_R_Squared'] + \
           [c for c in results_df.columns if c not in ['Factor', 'Coefficient (Slope)', 'P-Value', 'Model_R_Squared']]
    results_df = results_df[cols]

    results_df.to_csv(output_file, index=False)
    print(f"Analysis complete. Results saved to '{output_file}'.")

    # Print a quick summary to console
    print("\n--- Quick Summary ---")
    print(f"R-Squared: {r_squared:.4f}")
    print("Significant Factors (p < 0.05):")
    sig_factors = results_df[results_df['P-Value'] < 0.05]['Factor'].tolist()
    print(sig_factors if sig_factors else "None")

if __name__ == "__main__":
    # Replace 'your_data.csv' with the actual path to your file
    INPUT_CSV = 'output/station_data.csv'
    OUTPUT_CSV = 'output/spatial_analysis_simple.csv'

    run_spatial_regression(INPUT_CSV, OUTPUT_CSV)
