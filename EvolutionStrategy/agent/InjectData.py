import os
import pandas as pd
from config import mysql_config
import mysql.connector
from sqlalchemy import exc
# Define the directory containing the CSV files
directory = 'DataTraining'
# Establish connection to MySQL
try:
    conn = mysql.connector.connect(**mysql_config)
    cursor = conn.cursor()
    print("Connected to MySQL database")
except mysql.connector.Error as err:
    print(f"Error: {err}")
    exit(1)
# Get a list of all CSV files in the directory
csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

for file in csv_files:
    file_path = os.path.join(directory, file)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    symbol = file.split(' ')[0]
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
    
    # Convert 'Close' and 'Volume' columns to double (float) format
    if 'Close' in df_cleaned.columns and df_cleaned['Close'].dtype == 'object':
        df_cleaned['Close'] = df_cleaned['Close'].str.replace(',', '').astype(float)
    if 'High' in df_cleaned.columns and df_cleaned['High'].dtype == 'object':
        df_cleaned['High'] = df_cleaned['High'].str.replace(',', '').astype(float)
    if 'Low' in df_cleaned.columns and df_cleaned['Low'].dtype == 'object':
        df_cleaned['Low'] = df_cleaned['Low'].str.replace(',', '').astype(float)
    if 'Open' in df_cleaned.columns and df_cleaned['Open'].dtype == 'object':
        df_cleaned['Open'] = df_cleaned['Open'].str.replace(',', '').astype(float)   
    
    # Sort the DataFrame by the 'Date' column
    df_sorted = df_cleaned.sort_values(by='Date')

    
    # Check if the symbol already exists in the StockInfor table
    cursor.execute("SELECT id FROM stockinfors WHERE Symbol = %s", (symbol,))
    result = cursor.fetchone()
    if result:
        stock_infor_id = result[0]
    else:
        # Insert the symbol into the StockInfor table
        cursor.execute("INSERT INTO stockinfors (Symbol) VALUES (%s)", (symbol,))
        conn.commit()
        stock_infor_id = cursor.lastrowid

     # Add the StockInforId to the DataFrame
    df_sorted['StockInforId'] = stock_infor_id
    
    # Reorder DataFrame columns to match the StockData table
    df_sorted = df_sorted[['Date', 'Close', 'Open', 'High', 'Low', 'StockInforId']]    

    
    # Insert the data into the StockData table
    try:
        # Iterate over rows to insert data one by one
        for index, row in df_sorted.iterrows():
            cursor.execute("INSERT INTO stockdatas (Date, Close, Open, High, Low, StockInforId) VALUES (%s, %s, %s, %s, %s, %s)",
                           (row['Date'], row['Close'], row['Open'], row['High'], row['Low'], row['StockInforId']))
        conn.commit()
        print(f"Inserted data for symbol: {symbol} from file: {file_path}")
    except exc.SQLAlchemyError as e:
        print(f"Error inserting data for {symbol}: {e}")

# Close the cursor and connection
cursor.close()
conn.close()
    
    # Save the cleaned and sorted DataFrame back to CSV
df_sorted.to_csv(file_path, index=False)

print(f"Processed file: {file_path}")