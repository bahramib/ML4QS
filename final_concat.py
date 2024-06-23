import pandas as pd

# Paths to your files

# bike_paths = ['bike1/conv_and_frequency_bike1_data.csv']
# run_paths = ['running1/conv_and_frequency_running1_data.csv', 'running2/conv_and_frequency_running2_data.csv', 'running3/conv_and_frequency_running3_data.csv']
# walk_paths = ['walk1/conv_and_frequency_walk1_data.csv']
# dancing_paths = ['dancing2/conv_and_frequency_dancing2_data.csv']

matei_paths = ['walk_matei1/conv_and_frequency_walk_matei1_data.csv', 'walk_matei2/conv_and_frequency_walk_matei2_data.csv']
stan_paths = ['walk_stan1/conv_and_frequency_walk_stan1_data.csv', 'walk_stan2/conv_and_frequency_walk_stan2_data.csv', ]
beni_paths = ['walk_beni1/conv_and_frequency_walk_beni1_data.csv', 'walk_beni2/conv_and_frequency_walk_beni2_data.csv', ]

test_paths = ['walk_matei3/conv_and_frequency_walk_matei3_data.csv', 'walk_stan3/conv_and_frequency_walk_stan3_data.csv', 'walk_beni3/conv_and_frequency_walk_beni3_data.csv']

test_data = pd.read_csv(test_paths[0]).assign(person="matei")
test_data = pd.concat([test_data, pd.read_csv(test_paths[1]).assign(person="stan")])
test_data = pd.concat([test_data, pd.read_csv(test_paths[2]).assign(person="beni")])



# Load and label each dataset

# bike_data = pd.concat([pd.read_csv(file).assign(sport_type='bike') for file in bike_paths])
# run_data = pd.concat([pd.read_csv(file).assign(sport_type='run') for file in run_paths])
# walk_data = pd.concat([pd.read_csv(file).assign(sport_type='walk') for file in walk_paths])
# dancing_data = pd.concat([pd.read_csv(file).assign(sport_type='dance') for file in dancing_paths])

matei_data = pd.concat([pd.read_csv(file).assign(person='matei') for file in matei_paths])
stan_data = pd.concat([pd.read_csv(file).assign(person='stan') for file in stan_paths])
beni_data = pd.concat([pd.read_csv(file).assign(person='beni') for file in beni_paths])

# Combine into a single dataframe
all_data = pd.concat([matei_data, stan_data, beni_data])
# all_data = pd.concat([run_data, walk_data])

all_data.to_csv("final_data.csv", index=False)
test_data.to_csv("final_test.csv", index=False)

# Optionally shuffle the data if you want to randomize the order
all_data = all_data.sample(frac=1).reset_index(drop=True)
all_data.to_csv("final_data_shuffled.csv", index=False)
