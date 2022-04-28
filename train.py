import pandas as pd 
from autogluon.tabular import TabularPredictor
import json
import os  

save_path = "./models"
data_path = os.path.join("data")
train = pd.read_csv(os.path.join(data_path,'train.csv'))

predictor = TabularPredictor(label = "Survived", path = save_path).fit(train_data = train,presets='best_quality', time_limit=40)
train_acc = predictor.evaluate(train)

# print to file 
with open("metrics.json","w") as outfile:
    json.dump(train_acc,outfile)

