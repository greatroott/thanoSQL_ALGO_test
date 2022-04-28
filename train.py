import pandas as pd 
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
import json
import os  

save_path = "./models"
data_path = os.path.join("data")
df = pd.read_csv(os.path.join(data_path,'train.csv'))

train,test = train_test_split(df, test_size=0.2, shuffle=True, random_state=34)
predictor = TabularPredictor(label = "Survived", path = save_path).fit(train_data = train,presets='best_quality', time_limit=40)
acc = predictor.evaluate(test)

# print to file 
with open("metrics.json","w") as outfile:
    json.dump(acc,outfile)


