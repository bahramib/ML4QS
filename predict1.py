import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

shuffle = True


data = pd.read_csv('final_data.csv' if shuffle == False else 'final_data_shuffled.csv')

X = data.drop('person', axis=1)
X = X.drop('Long', axis=1)
X = X.drop('Lat', axis=1)
X = X.drop('Height', axis=1)
X = X.drop('Interval', axis=1)

y = data['person']


X.fillna(X.mean(), inplace=True)

# Check class distribution
print("Class distribution:\n", y.value_counts())

# average_size = int((y.value_counts().sum()) / 4)
# average_size = int((y.value_counts().sum()) / 2)
# over_strategy = {'matei': average_size, 'stan': average_size}
# under_strategy = {'matei': average_size, 'stan': average_size}
# over = RandomOverSampler(sampling_strategy=over_strategy)
# under = RandomUnderSampler(sampling_strategy=under_strategy)
# pipeline = Pipeline(steps=[('o', over), ('u', under)])


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# X_resampled, y_resampled = pipeline.fit_resample(X, y)
# print(pd.Series(y_resampled).value_counts())

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=69)


knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=10)  # 10-fold cross-validation
print("Cross-validated accuracy scores:", scores)
print("Average cross-validation score:", scores.mean())


knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=["matei", "stan", "beni"])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["matei", "stan", "beni"], yticklabels=["matei", "stan", "beni"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()