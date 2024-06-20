import os
import pandas as pd
import matplotlib.pyplot as plt


data1 = pd.read_csv("walk_matei3/Accelerometer.csv")
data2 = pd.read_csv("walk_matei3/Accelerometer.csv")
#plot time against acceleration x then y then z
plt.figure(figsize=(10, 6))
plt.plot(data1["Time (s)"], data1["X (m/s^2)"], label="X1")
# plt.plot(data2["Time (s)"], data2["X (m/s^2)"], label="X2")
plt.title("Accelerometer")
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.legend()
plt.show()


# running_data = pd.read_csv("running2/conv_and_frequency_running2_data.csv")
# walk_data = pd.read_csv("walk1/conv_and_frequency_walk1_data.csv")
# dancing_data = pd.read_csv("dancing2/conv_and_frequeency_dancing2_data.csv")
# bike_data = pd.read_csv("bike1/conv_and_frequency_bike1_data.csv")

# upper_limit = 500

#only plot up to time upper limit
# running_data = running_data[running_data["Interval"] <= upper_limit]
# walk_data = walk_data[walk_data["Interval"] <= upper_limit]
# dancing_data = dancing_data[dancing_data["Interval"] <= upper_limit]
# bike_data = bike_data[bike_data["Interval"] <= upper_limit]

# plt.figure(figsize=(10, 6))
# plt.plot(running_data["Interval"], running_data["A_X_temp_mean_ws_120"], label="Running")
# plt.plot(walk_data["Interval"], walk_data["A_X_temp_mean_ws_120"], label="Walking")
# plt.plot(dancing_data["Interval"], dancing_data["A_X_temp_mean_ws_120"], label="Dancing")
# plt.plot(bike_data["Interval"], bike_data["A_X_temp_mean_ws_120"], label="Biking")
# plt.title("Acceleration_X aggregated mean")
# plt.xlabel('Time (s)')
# plt.ylabel('Acceleration (m/s^2)')
# plt.legend()
# plt.show()

#put 3 subplots in one figure
# fig, axs = plt.subplots(3, figsize=(10, 6))
# fig.suptitle('Acceleration_X aggregated frequency data')
# axs[0].plot(running_data["Interval"], running_data["A_X_max_freq"], label="Running")
# axs[0].plot(walk_data["Interval"], walk_data["A_X_max_freq"], label="Walking")
# axs[0].plot(dancing_data["Interval"], dancing_data["A_X_max_freq"], label="Dancing")
# axs[0].plot(bike_data["Interval"], bike_data["A_X_max_freq"], label="Biking")
# axs[1].plot(running_data["Interval"], running_data["A_X_freq_weighted"], label="Running")
# axs[1].plot(walk_data["Interval"], walk_data["A_X_freq_weighted"], label="Walking")
# axs[1].plot(dancing_data["Interval"], dancing_data["A_X_freq_weighted"], label="Dancing")
# axs[1].plot(bike_data["Interval"], bike_data["A_X_freq_weighted"], label="Biking")
# axs[2].plot(running_data["Interval"], running_data["A_X_pse"], label="Running")
# axs[2].plot(walk_data["Interval"], walk_data["A_X_pse"], label="Walking")
# axs[2].plot(dancing_data["Interval"], dancing_data["A_X_pse"], label="Dancing")
# axs[2].plot(bike_data["Interval"], bike_data["A_X_pse"], label="Biking")
# axs[0].set_title("Max frequency")
# axs[0].set_xlabel('Time (s)')
# axs[0].set_ylabel('Frequency (Hz)')
# axs[0].legend()
# axs[1].set_title("Frequency weighted")
# axs[1].set_xlabel('Time (s)')
# axs[1].set_ylabel('Frequency (Hz)')
# axs[1].legend()
# axs[2].set_title("PSE")
# axs[2].set_xlabel('Time (s)')
# axs[2].set_ylabel('PSE')
# axs[2].legend()

# plt.show()




