import scipy.io
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

mat_data = scipy.io.loadmat('wheat.mat')
calibration_X = pd.DataFrame(mat_data['Calibration_X'])
calibration_Y = pd.DataFrame(mat_data['Calibration_Y'])
validation_X = pd.DataFrame(mat_data['Validation_X'])
validation_Y = pd.DataFrame(mat_data['Validation_Y'])
calibration_data = pd.concat([calibration_X, calibration_Y], axis=1)
validation_data = pd.concat([validation_X, validation_Y], axis=1)
combined_data = pd.concat([calibration_data, validation_data], axis=0)
combined_data.to_csv('wheat.csv', index=False)
scaler = MinMaxScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(combined_data), columns=combined_data.columns)
normalized_data.to_csv('wheat2.csv', index=False)
print(len(normalized_data.columns))