import pandas as pd

xlsx = pd.read_excel("C:/Users/R-Yoon/Downloads/Time-Series/Stock_Data.xlsx")
xlsx.to_csv("C:/Users/R-Yoon/Downloads/Time-Series/Stock_Data.csv")
