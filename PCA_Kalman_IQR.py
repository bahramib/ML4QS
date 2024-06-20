import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import os

directory = 'walk_beni3'
file_name = 'compressed_0.25_60_60.csv'
file_path = os.path.join(directory, file_name)
data = pd.read_csv(file_path)

features = ['LA_X', 'LA_Y', 'LA_Z', 'A_X', 'A_Y', 'A_Z', 'G_X', 'G_Y', 'G_Z', 
            'Lat', 'Long', 'Height', 'V', 'Dir', 'Hor_Acc', 'Vert_Acc']
X = data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

cov_matrix = np.cov(X_pca, rowvar=False)
inv_cov_matrix = np.linalg.inv(cov_matrix)
mean_distr = X_pca.mean(axis=0)

def mahalanobis_distance(x, mean_distr, inv_cov_matrix):
    return mahalanobis(x, mean_distr, inv_cov_matrix)

m_distances = np.apply_along_axis(mahalanobis_distance, 1, X_pca, mean_distr, inv_cov_matrix)

threshold = np.percentile(m_distances, 95)
outliers_pca = m_distances > threshold

kf = KalmanFilter(initial_state_mean=X_scaled[0],
                  n_dim_obs=X_scaled.shape[1])

state_means, _ = kf.smooth(X_scaled)

residuals = X_scaled - state_means
residuals_mean = np.mean(residuals, axis=0)
residuals_cov = np.cov(residuals, rowvar=False)
inv_residuals_cov = np.linalg.inv(residuals_cov)

residuals_m_distances = np.apply_along_axis(mahalanobis_distance, 1, residuals, residuals_mean, inv_residuals_cov)

residuals_threshold = np.percentile(residuals_m_distances, 95)
outliers_kalman = residuals_m_distances > residuals_threshold

outliers_combined = np.logical_or(outliers_pca, outliers_kalman)

cleaned_data = data[~outliers_combined]

def detect_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_mask = (df < lower_bound) | (df > upper_bound)
    return outlier_mask.any(axis=1)

extreme_outliers = detect_outliers_iqr(cleaned_data)
cleaned_data_final = cleaned_data[~extreme_outliers]

print(cleaned_data_final.head())

cleaned_data_final.to_csv("cleaned_compressed_outlier_removal.csv", index=False)

print(f"Outliers removed: {outliers_combined.sum() + extreme_outliers.sum()}")
print(f"Cleaned data saved to: cleaned_compressed_outlier_removal.csv")

plt.figure(figsize=(15, 10))
cleaned_data_final[features].boxplot(vert=False)
plt.title('Box Plot of All Features in the Cleaned Data')
plt.xlabel('Values')
plt.ylabel('Features')
plt.grid(True)
plt.tight_layout()
plt.show()
