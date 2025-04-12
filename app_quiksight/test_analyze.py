
import pandas as pd
from services.data_overview import get_data_overview
from services.missing_data_analysis import analyze_missing_data

# Use a relative path for the test data
df = pd.read_csv("C:/Users/HP/Downloads/matches.csv")  # adjust this path as needed

overview = get_data_overview(df)
missing = analyze_missing_data(df)

print("=== OVERVIEW ===")
print(overview)

print("\n=== MISSING DATA ===")
print(missing)







