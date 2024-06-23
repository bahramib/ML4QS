import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from scikeras.wrappers import KerasClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



# Load and prepare your data
data = pd.read_csv('final_data.csv')
X = data.drop(['person'], axis=1)
X = X.drop('Long', axis=1)
X = X.drop('Lat', axis=1)
X = X.drop('Height', axis=1)
X.fillna(X.mean(), inplace=True)
X = X.drop('Interval', axis=1).values
y = pd.get_dummies(data['person']).values

test_data = pd.read_csv("final_test.csv")
X_test = test_data.drop(['person'], axis=1)
X_test = X_test.drop('Long', axis=1)
X_test = X_test.drop('Lat', axis=1)
X_test = X_test.drop('Height', axis=1)
X_test.fillna(X_test.mean(), inplace=True)
X_test = X_test.drop('Interval', axis=1).values
y_test = pd.get_dummies(test_data['person']).values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.fit_transform(X_test)

# Reshape input to be [samples, time steps, features]
X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))
X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

X_train = X_scaled
y_train = y
X_test = X_test_scaled




# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=69)


def create_model(lstm_units=50, dropout_rate=0.2, optimizer='adam', loss='categorical_crossentropy'):
    model = Sequential()
    model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(1, X_train.shape[2])))
    model.add(Dropout(rate=dropout_rate))
    model.add(LSTM(units=lstm_units))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

# model = KerasClassifier(build_fn=create_model, verbose=0)
# print(model.get_params().keys())


# param_grid = {
#     # 'lstm_units': [50, 100, 150],
#     # 'dropout_rate': [0.1, 0.2, 0.3],
#     'optimizer': ['adam', 'rmsprop'],
#     'loss': ['categorical_crossentropy', 'sparse_categorical_crossentropy'],
#     'batch_size': [32, 64, 128],
#     'epochs': [10, 20, 30, 40]
# }


# grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)
# grid_result = grid.fit(X_train, y_train)


# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# for mean, stdev, param in zip(grid_result.cv_results_['mean_test_score'], grid_result.cv_results_['std_test_score'], grid_result.cv_results_['params']):
#     print("%f (%f) with: %r" % (mean, stdev, param))

lstm_unitss = [50, 100, 150]
dropout_rates = [0.1, 0.2, 0.3]

best = 0

# for lstm_units in lstm_unitss:
#     for dropout_rate in dropout_rates:
#         model = create_model(lstm_units=lstm_units, dropout_rate=dropout_rate, optimizer="rmsprop", loss="categorical_crossentropy")
#         history = model.fit(X_train, y_train, epochs=40, batch_size=128, validation_data=(X_test, y_test), verbose=1)
#         results = model.evaluate(X_test, y_test, verbose=0)
#         print("Test Accuracy: {:.2f}%".format(results[1] * 100))
#         with open("tune.txt", "a") as ff:
#             ff.write(f"Accuracy: {results[1] * 100}\n")
#             ff.write(f"lstm_units: {lstm_units}\tdropout_rate: {dropout_rate}\n\n\n")
#         ff.close()
#         if results[1] * 100 > best:
#             best = results[1] * 100
#             with open("best.txt", "w") as gg:
#                 gg.write(f"Accuracy: {results[1] * 100}\n")
#                 gg.write(f"lstm_units: {lstm_units}\tdropout_rate: {dropout_rate}")
#             gg.close()
            
            
model = create_model(lstm_units=150, dropout_rate=0.3, optimizer="rmsprop", loss="categorical_crossentropy")
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=40, batch_size=128, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
results = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy: {:.2f}%".format(results[1] * 100))


y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['matei', 'stan', 'beni'], yticklabels=['matei', 'stan', 'beni'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for LSTM Model')
plt.show()
