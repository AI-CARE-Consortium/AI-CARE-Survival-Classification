import pandas as pd
import glob

def consistency_check(registry:str):
    dataframes = []
    months = [6, 12, 18, 24]

    for i, file in enumerate([f"./results/predictions_{registry}_{month}.csv" for month in months]):
        dataframes.append(pd.read_csv(file, index_col="Patient_ID_unique", sep=",", dtype={"Patient_ID_unique": str, "prediction": bool, f"deathwithin{months[i]}months": bool}))

    for i in range(len(dataframes)):
        for j in range(i+1, len(dataframes)):
            merged = pd.merge(dataframes[i], dataframes[j], on="Patient_ID_unique", how="inner", suffixes=("_"+str(months[i]), "_"+str(months[j])))

            inconsistencies = merged[merged[f"prediction_{months[i]}"] & ~merged[f"prediction_{months[j]}"]]
            print(inconsistencies)

if __name__ == "__main__":
    consistency_check("1")

    

    