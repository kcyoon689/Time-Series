import pandas as pd
import csv

# output_csv=open('C:/Users/R-Yoon/Downloads/Time-Series/household_power_consumption.csv', "w")
# we = csv.writer(output_csv)

# file = pd.read_csv('C:\Users\R-Yoon\Downloads\Time-Series\household_power_consumption\household_power_consumption.txt', delimiter = ';')
file = pd.read_csv('C:/Users/R-Yoon/Downloads/Time-Series/household_power_consumption/household_power_consumption.txt', delimiter = ';')


# file = pd.read_csv(r'C:\Users\tonyloi\Desktop\sample.txt')
# new_csv_file = file.to_csv(r'C:/Users/R-Yoon/Downloads/Time-Series/household_power_consumption.csv', "w")
new_csv_file = file.to_csv(r'C:/Users/R-Yoon/Downloads/Time-Series/household_power_consumption.xlsx')
