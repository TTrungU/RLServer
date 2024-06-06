import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
def Detection(data):
    data['Close'] = data['Close'].fillna(data['forecast'])
    outliers_fraction = float(.05)
    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    # Create a copy of the original DataFrame to keep other columns intact
    data_scaled = data.copy()
    data_scaled['Close'] = np_scaled
    
    model = IsolationForest(contamination=outliers_fraction)
    model.fit(data_scaled[['Close']])
    data_scaled['anomaly'] = model.predict(data_scaled[['Close']])
    
    # Preserve the original columns including 'Date'
    data['anomaly'] = data_scaled['anomaly']
    
    return data