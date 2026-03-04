import pandas as pd
import numpy as np
import sklearn as sk
import seaborn as sns
import kagglehub
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


#CONNECTING KAGGLE
path = kagglehub.dataset_download("nickcantalupa/nfl-team-data-2003-2023")
#print("Path to dataset files:", path)
#print(os.listdir(path))


#FILE PATH
file_path = os.path.join(path, "team_stats_2003_2023.csv")
data = pd.read_csv(file_path)


#CLEANING DATA
data['mov'] = data['points_diff'] / data['g']
data.fillna({'ties': 0}, inplace=True)
#display(data)
#val = data.isnull().values.any()
#print(val)


#PFF EQUATION
offense = (data['yds_per_play_offense'], data['turnovers'], data['first_down'], (data['pass_cmp'] / data['pass_att']), data['pass_yds'], data['pass_net_yds_per_att'], data['rush_yds_per_att'], data['penalties'], data['penalties_yds'])
#display(offense)

defense = data['points_opp']
#display(defense)


#GRAPHS
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Correlation of Key Stats with Win-Loss Percentage (Training Data)', fontsize=16)

sns.regplot(x='yds_per_play_offense', y='win_loss_perc', data= data, ax=axes[0, 0], scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
axes[0, 0].set_title('Yards Per Play Offense vs. Win-Loss Percentage')
axes[0, 0].set_xlabel('Yards Per Play Offense')
axes[0, 0].set_ylabel('Win-Loss Percentage')

sns.regplot(x='points_opp', y='win_loss_perc', data= data, ax=axes[0, 1], scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
axes[0, 1].set_title('Points Allowed vs. Win-Loss Percentage')
axes[0, 1].set_xlabel('Points Allowed')
axes[0, 1].set_ylabel('Win-Loss Percentage')

sns.regplot(x='turnovers', y='win_loss_perc', data= data, ax=axes[1, 0], scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
axes[1, 0].set_title('Turnovers vs. Win-Loss Percentage')
axes[1, 0].set_xlabel('Turnovers')
axes[1, 0].set_ylabel('Win-Loss Percentage')

sns.regplot(x='pass_net_yds_per_att', y='win_loss_perc', data= data, ax=axes[1, 1], scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
axes[1, 1].set_title('Net Pass Yards Per Attempt vs. Win-Loss Percentage')
axes[1, 1].set_xlabel('Net Pass Yards Per Attempt')
axes[1, 1].set_ylabel('Win-Loss Percentage')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


#LR MODEL
latest_season = data['year'].max()
train = data[data['year'] < latest_season].copy()
test = data[data['year'] == latest_season].copy()

offenseTrain = (train['yds_per_play_offense'], train['turnovers'], train['first_down'], (train['pass_cmp'] / train['pass_att']), train['pass_yds'], train['pass_net_yds_per_att'], train['rush_yds_per_att'], train['penalties'], train['penalties_yds'])
defenseTrain = train['points_opp']
offenseTest = (test['yds_per_play_offense'], test['turnovers'], test['first_down'], (test['pass_cmp'] / test['pass_att']), test['pass_yds'], test['pass_net_yds_per_att'], test['rush_yds_per_att'], test['penalties'], test['penalties_yds'])
defenseTest = test['points_opp']

xTrain = np.column_stack(offenseTrain + (defenseTrain,))
yTrain = train['win_loss_perc']
xTest = np.column_stack(offenseTest + (defenseTest,))
yTest = test['win_loss_perc']

model = LinearRegression()
model.fit(xTrain, yTrain)
predictionTrain= model.predict(xTrain)
predictionTest = model.predict(xTest)
r2 = r2_score(yTest, predictionTest)
mae = mean_absolute_error(yTest, predictionTest)
rmse = np.sqrt(mean_squared_error(yTest, predictionTest))


#print(model.intercept_)
#print(model.coef_)
print(f"R-squared: {r2}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
#print(predictionTest)
predictions_df = pd.DataFrame({
    'team': test['team'].reset_index(drop=True),
    'predicted_win_perc': predictionTest
})
display(predictions_df)