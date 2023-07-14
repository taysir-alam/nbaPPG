import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

data = pd.read_csv('nba_data.csv')

features = ['ppg', 'fg%', '3p%']
target = 'points_scored'

X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], random_state=0)

model = DecisionTreeRegressor(max_depth=3, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)


plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=features, rounded=True, fontsize=14)
plt.show()

new_player = [[20.0, 1, 108.0]]
predicted_points = model.predict(new_player)
print('Predicted points for new player:', predicted_points)
