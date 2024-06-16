import os
import pandas as pd
import matplotlib.pyplot as plt

directory = "dancing2"
file_pah = os.path.join(directory, "Accelerometer.csv")

if os.path.exists(file_pah):
    data = pd.read_csv(file_pah)
    #plot time against acceleration x then y then z
    plt.figure(figsize=(10, 6))
    plt.plot(data["Time (s)"], data["X (m/s^2)"], label="X")
    plt.plot(data["Time (s)"], data["Y (m/s^2)"], label="Y")
    plt.plot(data["Time (s)"], data["Z (m/s^2)"], label="Z")
    plt.title("Accelerometer")
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.show()

else:
    print(f"The file {file_pah} does not exist.")

