
#import classes from FrequencyAbstraction.py and TemporalAbstraction.py
from FrequencyAbstraction import FourierTransformation
from TemporalAbstraction import NumericalAbstraction
import os
import pandas as pd
import matplotlib.pyplot as plt

directory = "running3"
file_name = "cleaned_compressed_outlier_removal.csv"

# Create instances of the classes
fourier = FourierTransformation()
numerical = NumericalAbstraction()

# Full path to the file
file_path = os.path.join(directory, file_name)

# Check if the file exists
if os.path.exists(file_path):
    # Load the data
    data = pd.read_csv(file_path)
    # aggregated_data = pd.read_csv("aggregated_data.csv")
    # Apply numerical abstraction
    aggregated_data = numerical.abstract_numerical(data,["LA_X","LA_Y","LA_Z","A_X","A_Y","A_Z","G_X","G_Y","G_Z","Lat","Long","Height","V","Dir","Hor_Acc","Vert_Acc"], 120, "mean")
    
    # print aggregated data to file
    # aggregated_data.to_csv("aggregated_data.csv",index=False)
    
    # Plot each column on a single graph
    plt.figure(figsize=(10, 6))
    for column in aggregated_data.columns:
        if 'LA_X' in column:
            plt.plot(aggregated_data["Interval"], aggregated_data[column], label=column)
    plt.title("Accelerometer")
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.show()
    frequency_data = fourier.abstract_frequency(aggregated_data, ["LA_X","LA_Y","LA_Z","A_X","A_Y","A_Z","G_X","G_Y","G_Z"], 40, 4)

    frequency_data.to_csv(f"conv_and_frequeency_{directory}_data.csv",index=False)
    # frequency_data = pd.read_csv("frequeency_data.csv")
    
    plt.figure(figsize=(10, 6))
    for column in frequency_data.columns:
        if 'LA_X' in column and column != "LA_X_freq_weighted":
            plt.plot(frequency_data["Interval"], frequency_data[column], label=column)
    plt.title("Accelerometer")
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.show()

else:
    print(f"The file compressed.csv does not exist.")
