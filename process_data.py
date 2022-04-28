import pandas as pd 
import sweetviz
import os 
data_path = os.path.join("data")
train = pd.read_csv(os.path.join(data_path,'train.csv'))
# html file 생성
advert_report = sweetviz.analyze(train)
# display the report 
advert_report.show_html('./report_eda.html')