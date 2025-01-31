import os
import pandas as pd

def process_csv_files(directory='./'):
    # Get a list of all CSV files in the directory
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    
    for file in csv_files:
        file_path = os.path.join(directory, file)
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        symbol = file.split(' ')[0].split('.')[0]
        
        # Rename the 'Price' column to 'Close' if 'Price' exists
        if 'Price' in df.columns:
            df.rename(columns={'Price': 'Close'}, inplace=True)
        
        # Remove 'Vol.' and 'Change %' columns if they exist
        columns_to_remove = ['Vol.', 'Change %']
        df.drop(columns=[col for col in columns_to_remove if col in df.columns], inplace=True)
        
        # Drop rows with any null values
        df_cleaned = df.dropna()
        
        # Convert the 'Date' column to datetime format
        df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'])
        
        # Convert 'Close', 'High', 'Low', and 'Open' columns to float format
        float_columns = ['Close', 'High', 'Low', 'Open']
        for col in float_columns:
            if col in df_cleaned.columns and df_cleaned[col].dtype == 'object':
                df_cleaned[col] = df_cleaned[col].str.replace(',', '').astype(float)
        
        # Sort the DataFrame by the 'Date' column
        df_sorted = df_cleaned.sort_values(by='Date')
        
        # Save the cleaned and sorted DataFrame back to CSV
        df_sorted.to_csv(file_path, index=False)
        print(f"Successfully processed: {file}")

# Call the function
if __name__ == "__main__":
    process_csv_files()
