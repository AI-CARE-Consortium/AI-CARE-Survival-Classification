import pandas as pd
import glob

def consistency_check(registry:str):
    dataframes = []
    months = [6, 12, 18, 24]

    for i, file in enumerate([f"./results/predictions_{registry}_{month}.csv" for month in months]):
        dataframes.append(pd.read_csv(file, index_col="Patient_ID_unique", sep=",", dtype={"Patient_ID_unique": str, "prediction": bool, f"deathwithin{months[i]}months": bool}))
    inconsitent_patients = []
    for i in range(len(dataframes)):
        for j in range(i+1, len(dataframes)):
            merged = pd.merge(dataframes[i], dataframes[j], on="Patient_ID_unique", how="inner", suffixes=("_"+str(months[i]), "_"+str(months[j])))

            inconsistencies = merged[merged[f"prediction_{months[i]}"] & ~merged[f"prediction_{months[j]}"]]
            print(inconsistencies)
            inconsitent_patients.extend(inconsistencies.index)
    set_inconsitent_patients = set(inconsitent_patients)
   
    return set_inconsitent_patients
    


if __name__ == "__main__":
    all_inconsistent_patients = set()
    for registry in ["1", "2", "3", "5", "10", "14", "15"]:
        all_inconsistent_patients = all_inconsistent_patients.union(consistency_check(registry))

    with open(f"./results/inconsistencies.txt", "wt") as f:
        for patient in all_inconsistent_patients:
            f.write(patient + "\n")

    

    