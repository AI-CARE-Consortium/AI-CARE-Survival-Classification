from data_import.data_loading import import_aicare
from data_import.data_preprocessing import calculate_survival_time, calculate_outcome_in_X_years

from sklearn.feature_selection import RFECV
from catboost import CatBoostClassifier, Pool
#from xgboost import XGBClassifier
import argparse
import logging
import pandas as pd
import datetime
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--registry', type=str, help='registry number')  
    parser.add_argument('--months', type=int, help='Months to binary classify')
    args = parser.parse_args()
    registry = args.registry
    months = args.months
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = f"./results/log_study_registry_{registry}_months_{months}.txt"   

    logger.addHandler(logging.FileHandler(log_path, mode="w"))
    logger.info(f"Starting study for registry {registry} and binary classification for {months} months at the time {datetime.datetime.now()}")

    # Load data
    aicare = import_aicare(path="./aicare/aicare_gesamt/", tumor_entity="lung", registry="all")
    dataset = aicare["patient"].drop(columns=["Patient_ID"])
    dataset = dataset.merge(aicare["tumor"].drop(columns=["Register_ID_FK", "Tumor_ID", "Patient_ID_FK"]), on="Patient_ID_unique")
    dataset = dataset.merge(aicare["op"].drop(columns=["Register_ID_FK", "Patient_ID_FK", "Tumor_ID_FK", "OP_ID"]), on="Patient_ID_unique", how="left")
    dataset = dataset.merge(aicare["strahlentherapie"].drop(columns=["Register_ID_FK", "Patient_ID_FK", "Bestrahlung_ID", "Tumor_ID_FK"]), on="Patient_ID_unique", how="left")
    dataset.rename(columns={"Stellung_OP": "Stellung_OP_st"}, inplace=True)
    dataset = dataset.merge(aicare["systemtherapie"].drop(columns=["Register_ID_FK", "Patient_ID_FK", "SYST_ID", "Tumor_ID_FK"]), on="Patient_ID_unique", how="left")
    dataset.rename(columns={"Stellung_OP": "Stellung_OP_syst"}, inplace=True)
    dataset = dataset.groupby("Patient_ID_unique").first().reset_index()

    # Binarize data depending on survival time

    dataset["survival_time"] = calculate_survival_time(dataset, "Datum_Vitalstatus", "Diagnosedatum")

    dataset = dataset[dataset["survival_time"]>=0]
    dataset["Alter_bei_Diagnose"] = ((dataset['Diagnosedatum'] - dataset['Geburtsdatum']).dt.days // 365.25).astype(int)


    dead = (dataset["survival_time"] < 30* months) & (dataset["Verstorben"] == 1)
    dataset[f"deathwithin{months}months"] = dead.values
    alive = dataset["survival_time"] >= 30 * months

    
    dataset = dataset[dead | alive]

    
    #replace nan with -1 for categorical features only
    for column in dataset.select_dtypes(include=['category']).columns:
        dataset.loc[:,column] = dataset.loc[:,column].astype(pd.CategoricalDtype(categories=dataset[column].cat.categories.append(pd.Index(["-1"])), ordered=True))
        dataset.loc[:,column] = dataset.loc[:,column].fillna("-1")
    #print(cat_feature_ind)

    dataset_test = dataset[dataset["Register_ID_FK"] == registry]
    y_test = dataset_test[f"deathwithin{months}months"]

    drop_cols = ["Register_ID_FK", "Patient_ID_unique", "Diagnosedatum", "Geburtsdatum", "Beginn_Bestrahlung", "Beginn_SYST", "Datum_OP", 
                 "Datum_Vitalstatus", "Verstorben", "survival_time", "Anzahl_Tage_Diagnose_Tod", 
                "Todesursache_Grundleiden", "Todesursache_Grundleiden_Version", f"deathwithin{months}months",
                "Anzahl_Tage_Diagnose_ST", "Anzahl_Tage_Diagnose_OP", "Anzahl_Tage_Diagnose_SYST", "Inzidenzort", "Primaertumor_DCN",
                "Anzahl_Tage_SYST", "Anzahl_Tage_ST", "Primaertumor_ICD_Version", "pTNM_Version", "cTNM_Version", "Primaertumor_Morphologie_ICD_O_Version",
                "Primaertumor_Topographie_ICD_O_Version", "Zielgebiet_CodeVersion", "TNM_Version", "Weitere_Todesursachen", "Weitere_Todesursachen_Version",
                "Menge_OPS_version"]
    
    obj_cols = dataset.select_dtypes(include=['object']).columns
    drop_cols.extend(obj_cols)


    X_test = dataset_test.drop(columns=drop_cols)
    dataset_train = dataset[dataset["Register_ID_FK"] != registry]
    y_train = dataset_train[f"deathwithin{months}months"]
    X_train = dataset_train.drop(columns=drop_cols)
    

    cat_feature_ind = [X_train.columns.get_loc(c) for c in X_train.columns if X_train[c].dtype.name == 'category']

    print(cat_feature_ind)
    print(','.join(X_train.columns))
    print(','.join(X_test.columns))

    text_feature_ind = [X_train.columns.get_loc(c) for c in X_train.columns if X_train[c].dtype.name == 'string']
    for column in X_train.select_dtypes(include=['string']).columns:
        X_train.loc[:,column] = X_train.loc[:,column].fillna("-1")
        X_test.loc[:,column] = X_test.loc[:,column].fillna("-1")
    cat_feature_ind.extend(text_feature_ind)





    #X_train[obj_cols] = X_train[obj_cols].astype('str')
    #X_test[obj_cols] = X_test[obj_cols].astype('str')
    #text_feature_ind = [X_train.columns.get_loc(c) for c in obj_cols]
    
    # Preprocessing

    # # Feature selection
    # estimator = CatBoostClassifier()
    # selector = RFECV(estimator, step=1, cv=5)
    # selector = selector.fit(X_train, y_train)
    # print(selector.support_)
    # print(selector.ranking_)

    # Validation
    model = CatBoostClassifier(early_stopping_rounds=20,
                               task_type="GPU",
                               devices='cuda:0',
                               nan_mode="Min",
                               cat_features=cat_feature_ind,
                               custom_metric=['Logloss', 'F1', 'Accuracy'],
                               train_dir=f"./catboost_info/{registry}_{months}/",
                               border_count=254,
                               depth=6
                               )
    
    train_pool = Pool(X_train, y_train, cat_features=cat_feature_ind)
    eval_pool = Pool(X_test, y_test, cat_features=cat_feature_ind)
    model.fit(train_pool, eval_set=eval_pool, plot_file=f"./results/plot_{registry}_{months}.html")
    print(model.score(X_test, y_test))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test)[:,1])
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc}')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(f"./results/roc_{registry}_{months}.png", dpi=500)
    plt.close()
    
    
    prediction = model.predict(X_train)
    dataset_train["prediction"] = prediction
    dataset_train[['Patient_ID_unique', 'prediction', f"deathwithin{months}months"]].to_csv(f"./results/predictions_{registry}_{months}.csv", index=False)

    logger.info(f"Test Registry: {registry} - binary classification for {months} months")
    logger.info(f"Dataset shape: {dataset.shape}")
    logger.info(f"Train shape: {dataset_train.shape}")
    logger.info(f"Test shape: {dataset_test.shape}")

    logger.info(f"Model score: {model.score(X_test, y_test)}")
    logger.info(metrics.classification_report(y_test, model.predict(X_test)))
    logger.info(model.get_feature_importance(prettified=True).to_string())
    model.save_model(f"./results/model_{registry}_{months}.cbm")
    tree = model.plot_tree(tree_idx=0, pool=train_pool)
    tree.save(f"./results/tree_{registry}_{months}.gv")
    os.system(f"dot -Tpng ./results/tree_{registry}_{months}.gv -o ./results/tree_{registry}_{months}.png")
    # for i, name in enumerate(X_train.columns):
    #     model.calc_feature_statistics(train_pool, feature=i, plot=True, cat_feature_values=cat_feature_ind ,plot_file=f"./results/features/{name}_{registry}_{months}.png")
