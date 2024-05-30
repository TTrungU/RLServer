import mysql.connector
from pymongo import MongoClient,UpdateOne
from config import mysql_config
import config
from decimal import Decimal
from datetime import datetime
# Connect to MySQL
mysql_conn = mysql.connector.connect(**mysql_config)
# Connect to MongoDB
mongo_client = MongoClient(config.MONGO_URI)
mongo_db = mongo_client['StockDB']
mongo_collection = mongo_db['StockData']

def convert_data_types(data):
    if isinstance(data, Decimal):
        return float(data)
    if isinstance(data, datetime):
        return data.isoformat()
    if isinstance(data, dict):
        return {k: convert_data_types(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_data_types(i) for i in data]
    return data


try:
    with mysql_conn.cursor() as cursor:
        # Fetch data from stockinfors and stockdata
        cursor.execute("SELECT id, symbol, description FROM stockinfors")
        stockinfors_data = cursor.fetchall()

        cursor.execute("SELECT Date, Close, Open, High, Low, StockInforId FROM stockdatas")
        stockdata_data = cursor.fetchall()

        # Group stockdata by StockInforId
        grouped_stockdata = {}
        for stockdatas in stockdata_data:
            stock_infor_id = stockdatas[-1]  # Last element is StockInforId
            if stock_infor_id not in grouped_stockdata:
                grouped_stockdata[stock_infor_id] = []
            grouped_stockdata[stock_infor_id].append(stockdatas[:-1])  # Exclude StockInforId

        operations = []

        # Combine stockinfors and stockdata into MongoDB schema
        for stockinfo in stockinfors_data:
            stockinfo_id, symbol, description = stockinfo
            stockdatas = grouped_stockdata.get(stockinfo_id, [])

            # Prepare stockdata documents
            stockdata_list = []
            for stockdata in stockdatas:
                stockdata_dict = {
                    "Date": convert_data_types(stockdata[0]),
                    "Close": convert_data_types(stockdata[1]),
                    "Open": convert_data_types(stockdata[2]),
                    "High": convert_data_types(stockdata[3]),
                    "Low": convert_data_types(stockdata[4])
                }
                stockdata_list.append(stockdata_dict)

            # Create MongoDB document
            mongo_document = {
                "Stockinfor": {
                    "id": stockinfo_id,
                    "Symbol": symbol,
                    "Description": description
                },
                "stockData": stockdata_list
            }

            # Create update operation to upsert data into MongoDB
            operation = UpdateOne(
                {"Stockinfor.id": stockinfo_id},
                {"$set": mongo_document},
                upsert=True
            )
            operations.append(operation)

        # Perform bulk write operation
        if operations:
            mongo_collection.bulk_write(operations)

finally:
    mysql_conn.close()
    mongo_client.close()

print("Data migration complete!")