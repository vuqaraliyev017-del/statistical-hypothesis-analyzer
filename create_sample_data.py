import pandas as pd
import numpy as np

np.random.seed(42)
n = 120

df = pd.DataFrame({
    "gender": np.random.choice(["Male", "Female"], size=n),
    "department": np.random.choice(["Finance", "HR", "Sales"], size=n),
    "satisfaction_score": np.random.normal(75, 10, size=n).round(2),
    "salary": np.random.normal(1500, 300, size=n).round(2),
    "performance_score": np.random.normal(80, 8, size=n).round(2),
    "before_training": np.random.normal(65, 10, size=n).round(2),
})

df["after_training"] = df["before_training"] + np.random.normal(5, 7, size=n).round(2)

# Department effect example
df.loc[df["department"] == "Finance", "satisfaction_score"] += 4
df.loc[df["department"] == "Sales", "satisfaction_score"] -= 3

df.to_excel("sample_data.xlsx", index=False)
print("sample_data.xlsx yaradıldı.")