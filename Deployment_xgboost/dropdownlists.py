import pandas as pd

#import data csv
df = pd.read_csv('input_data/streeteasy.csv')

boroughs = sorted(list(df["borough"].unique()))
neighborhoods = sorted(list(df["neighborhood"].unique()))
submarkets = sorted(list(df["submarket"].unique()))
