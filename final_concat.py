import pandas as pd

# Paths to your files
bike_paths = ['bike1/conv_and_frequency_bike1_data.csv']
run_paths = ['running1/conv_and_frequency_running1_data.csv', 'running2/conv_and_frequency_running2_data.csv', 'running3/conv_and_frequency_running3_data.csv']
walk_paths = ['walk1/conv_and_frequency_walk1_data.csv']

# Load and label each dataset
bike_data = pd.concat([pd.read_csv(file).assign(sport_type='bike') for file in bike_paths])
run_data = pd.concat([pd.read_csv(file).assign(sport_type='run') for file in run_paths])
walk_data = pd.concat([pd.read_csv(file).assign(sport_type='walk') for file in walk_paths])

# Combine into a single dataframe
all_data = pd.concat([bike_data, run_data, walk_data])
# all_data = pd.concat([run_data, walk_data])

all_data.to_csv("final_data.csv", index=False)

# Optionally shuffle the data if you want to randomize the order
# all_data = all_data.sample(frac=1).reset_index(drop=True)
# all_data.to_csv("final_data_shuffled.csv", index=False)
