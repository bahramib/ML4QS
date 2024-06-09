import pandas as pd
import numpy as np

# Hyperparameters (for us)
interval_length = 0.01

def replace_nan_with_next_numpy(df):
    # Identify numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Convert only numeric parts of the DataFrame to numpy array for direct manipulation
    numeric_data = df[numeric_cols].values
    
    # Iterate backwards through rows of the numeric data
    for i in range(len(numeric_data) - 2, -1, -1):
        # Find NaN elements in the current row
        nan_mask = np.isnan(numeric_data[i])
        if nan_mask.all():  # Check if all values in the row are NaN
            # Replace NaNs with values from the next row, but only for NaN locations
            numeric_data[i, nan_mask] = numeric_data[i + 1, nan_mask]

    # Assign the modified columns back to the DataFrame
    df[numeric_cols] = numeric_data
    return df

def fill(df_longer, df_shorter, prefix):
    if len(df_longer["Interval"]) == len(df_shorter["Interval"]): # I know this is a double check but still
        return df_shorter

    buffer = df_shorter
    # print(f'{len(df_shorter["Interval"]) + 1} to {len(df_longer["Interval"]) + 1}')
    for i in range(len(df_shorter["Interval"]), len(df_longer["Interval"])):
        # print(f"i. row")
        new_row = {'Interval': df_longer["Interval"][i], f'{prefix}_X': np.nan, f'{prefix}_Y': np.nan, f'{prefix}_Z': np.nan}
        buffer.loc[i] = new_row # buffer.append(new_row, ignore_index=True)
    return buffer

def fill_loc(df_longer, df_shorter, prefix):
    if len(df_longer["Interval"]) == len(df_shorter["Interval"]): # I know this is a double check but still
        return df_shorter

    buffer = df_shorter
    # print(f'{len(df_shorter["Interval"]) + 1} to {len(df_longer["Interval"]) + 1}')
    for i in range(len(df_shorter["Interval"]), len(df_longer["Interval"])):
        # print(f"i. row")
        new_row = {'Interval': df_longer["Interval"][i], 'Lat': np.nan, 'Long': np.nan, 'Height': np.nan, 'V': np.nan, 'Dir': np.nan, 'Hor_Acc': np.nan, 'Vert_Acc': np.nan}
        buffer.loc[i] = new_row # buffer.append(new_row, ignore_index=True)
    return buffer


# Load the files with the correct names
data_lin_acc = pd.read_csv('bike/Linear Accelerometer.csv')
data_acc = pd.read_csv('bike/Accelerometer.csv')
data_gyro = pd.read_csv('bike/Gyroscope.csv')
data_loc = pd.read_csv('bike/Location.csv')
data_lin_acc.columns = ["LA_Time", "LA_X", "LA_Y", "LA_Z"]
data_acc.columns = ["A_Time", "A_X", "A_Y", "A_Z"]
data_gyro.columns = ["G_Time", "G_X", "G_Y", "G_Z"]
data_loc.columns = ["Loc_Time", "Lat", "Long", "Height", "V", "Dir", "Hor_Acc", "Vert_Acc"]
# Cheat sheet:
# "Time","Latitude (°)","Longitude (°)","Height","Velocity","Direction (°)","Horizontal Accuracy (m)","Vertical Accuracy (°)"

# Grouping everything by interval with averaging as the aggregation (sorry for the WET code lol)
bins = np.arange(0, data_lin_acc["LA_Time"].max() + interval_length, interval_length)
data_lin_acc['Interval'] = pd.cut(data_lin_acc['LA_Time'], bins, right=False, include_lowest=True)
data_lin_acc = data_lin_acc.groupby('Interval', observed=False).mean().reset_index()
data_lin_acc['Interval'] = data_lin_acc['Interval'].apply(lambda x: x.left + interval_length)
data_lin_acc = data_lin_acc.drop('LA_Time', axis=1)

bins = np.arange(0, data_acc["A_Time"].max() + interval_length, interval_length)
data_acc['Interval'] = pd.cut(data_acc['A_Time'], bins, right=False, include_lowest=True)
data_acc = data_acc.groupby('Interval', observed=False).mean().reset_index()
data_acc['Interval'] = data_acc['Interval'].apply(lambda x: x.left + interval_length)
data_acc = data_acc.drop('A_Time', axis=1)

bins = np.arange(0, data_gyro["G_Time"].max() + interval_length, interval_length)
data_gyro['Interval'] = pd.cut(data_gyro['G_Time'], bins, right=False, include_lowest=True)
data_gyro = data_gyro.groupby('Interval', observed=False).mean().reset_index()
data_gyro['Interval'] = data_gyro['Interval'].apply(lambda x: x.left + interval_length)
data_gyro = data_gyro.drop('G_Time', axis=1)

data_loc = data_loc.fillna(0)

bins = np.arange(0, data_loc["Loc_Time"].max() + interval_length, interval_length)
data_loc['Interval'] = pd.cut(data_loc['Loc_Time'], bins, right=False, include_lowest=True)
data_loc = data_loc.groupby('Interval', observed=False).mean().reset_index()
data_loc = replace_nan_with_next_numpy(data_loc)
data_loc['Interval'] = data_loc['Interval'].apply(lambda x: x.left + interval_length)
data_loc = data_loc.drop('Loc_Time', axis=1)

# print(data_loc)


# Making sure every interval column has the same length so we can use merge instead of concat
dfs = [data_lin_acc, data_acc, data_gyro, data_loc]
longest_idx = np.argmax([len(data_lin_acc), len(data_acc), len(data_gyro), len(data_loc)])
first_valids = []
print(f"Longest index: {longest_idx}; longest df: {['Linear Accelerometer', 'Accelerometer', 'Gyroscope'][longest_idx]}; Length: {len(dfs[longest_idx])}")

if len(data_lin_acc) < len(dfs[longest_idx]):
    # print(len(data_lin_acc))
    data_lin_acc = fill(dfs[longest_idx], data_lin_acc, "LA")
    # print(len(data_lin_acc))
if len(data_acc) < len(dfs[longest_idx]):
    # print(len(data_acc))
    data_acc = fill(dfs[longest_idx], data_acc, "A")
    # print(len(data_acc))
if len(data_gyro) < len(dfs[longest_idx]):
    # print(len(data_gyro))
    data_gyro = fill(dfs[longest_idx], data_gyro, "G")
    # print(len(data_gyro))
if len(data_loc) < len(dfs[longest_idx]):
    # print(len(data_loc))
    data_loc = fill_loc(dfs[longest_idx], data_loc, "Loc")
    # print(len(data_loc))

# print(data_loc)

# data_gyro.drop("Interval")

# print("Same length" if len(data1) == len(data2) else "Yuck")
# Put the data together
# data = pd.concat([data_lin_acc, data_acc], axis=1, join="outer")
data = pd.merge(data_lin_acc, data_acc, on='Interval', how='outer')
data = pd.merge(data, data_gyro, on='Interval', how='outer')
data = pd.merge(data, data_loc, on='Interval', how='outer')


# missing_any = data.isna().any().any()
# print(missing_any)

# Show the resulting DataFrame
print(data)
data.to_csv('bike/compressed.csv', index=False)