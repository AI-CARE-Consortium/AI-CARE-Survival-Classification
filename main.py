from sklearn.dummy import DummyClassifier
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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from os import path
import optuna
import pathlib

if __name__ == '__main__':
    random_state = 21
    parser = argparse.ArgumentParser(description='Train a CatBoost model for binary classification')
    parser.add_argument('--data_path', type=str, help='Path to the data')
    parser.add_argument('--registry', type=str, help='registry number')  
    parser.add_argument('--months', type=int, help='Months to binary classify')
    parser.add_argument("--inverse", action="store_true", help="Inverse the binary classification")
    parser.add_argument("--dummy", action="store_true", help="Use dummy classifier that predicts the most frequent class")
    parser.add_argument("--entity", type=str, help="Entity to train on")
    args = parser.parse_args()
    registry = args.registry
    months = args.months
    
    entity = args.entity
    if entity not in ["lung", "thyroid", "breast", "non_hodgkin_lymphoma"]:
        raise ValueError("Entity must be one of lung, thyroid, breast, non_hodgkin_lymphoma")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if args.dummy:
        log_path = f"./results/dummy/log_study_registry_{registry}_months_{months}"
        dummy = "dummy/"
        if not path.exists("./results/dummy"):
            pathlib.Path("./results/dummy").mkdir(parents=True, exist_ok=True)
    else:
        log_path = f"./results/log_study_registry_{registry}_months_{months}"
        dummy = ""
    if args.inverse:
        log_path += "_inverse.txt"
    else:
        log_path += ".txt"
    study_db="sqlite:///optuna.db"
    logger.addHandler(logging.FileHandler(log_path, mode="w"))
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.
    logger.info(f"Starting study for registry {registry}, entity {entity} and binary classification for {months} months at the time {datetime.datetime.now()}")
    if args.inverse:
        logger.info("Inverse binary classification: 0 if patient dies within the time frame, 1 otherwise")
    else:
        logger.info("Binary classification: 1 if patient dies within the time frame, 0 otherwise")
    # Load data
    data_path = args.data_path
    print(data_path)
    
    if path.exists(f"{data_path}dataset_{months}_{entity}.pkl"):
        dataset = pd.read_pickle(f"{data_path}dataset_{months}_{entity}.pkl")
    else:
        aicare = import_aicare(path=data_path, tumor_entity=entity, registry=["1", "2", "3", "5", "10", "14", "15"])
        dataset = aicare["patient"].drop(columns=["Patient_ID"])
        # Merge all tables
        dataset = dataset.merge(aicare["tumor"].groupby("Patient_ID_unique").first().drop(columns=["Register_ID_FK", "Tumor_ID", "Patient_ID_FK"]), on="Patient_ID_unique")
        dataset = dataset.merge(aicare["op"].groupby("Patient_ID_unique").first().drop(columns=["Register_ID_FK", "Patient_ID_FK", "Tumor_ID_FK", "OP_ID"]), on="Patient_ID_unique", how="left")
        dataset = dataset.merge(aicare["strahlentherapie"].groupby("Patient_ID_unique").first().drop(columns=["Register_ID_FK", "Patient_ID_FK", "Bestrahlung_ID", "Tumor_ID_FK"]), on="Patient_ID_unique", how="left")
        dataset.rename(columns={"Stellung_OP": "Stellung_OP_st"}, inplace=True)
        dataset = dataset.merge(aicare["systemtherapie"].groupby("Patient_ID_unique").first().drop(columns=["Register_ID_FK", "Patient_ID_FK", "SYST_ID", "Tumor_ID_FK"]), on="Patient_ID_unique", how="left")
        dataset.rename(columns={"Stellung_OP": "Stellung_OP_syst"}, inplace=True)
        if entity == "breast":
            dataset = dataset.merge(aicare["modul_mamma"].drop(columns=["Register_ID_FK", "Patient_ID_FK", "Tumor_ID_FK"]), on="Patient_ID_unique", how="left")
        dataset = dataset.reset_index(drop=True)
        
        # Binarize data depending on survival time

        dataset["survival_time"] = calculate_survival_time(dataset, "Datum_Vitalstatus", "Diagnosedatum")

        dataset = dataset[dataset["survival_time"]>0]
        #dataset["Alter_bei_Diagnose"] = ((dataset['Diagnosedatum'] - dataset['Geburtsdatum']).dt.days // 365.25).astype(int)


        dead = (dataset["survival_time"] < 30* months) & (dataset["Verstorben"] == 1)
        dataset[f"deathwithin{months}months"] = dead.values
        alive = dataset["survival_time"] >= 30 * months
        dataset[f"liveslongerthan{months}months"] = alive.values
        
        dataset = dataset[dead | alive]

        
        #replace nan with -1 for categorical features only
        for column in dataset.select_dtypes(include=['category']).columns:
            dataset.loc[:,column] = dataset.loc[:,column].astype(pd.CategoricalDtype(categories=dataset[column].cat.categories.append(pd.Index(["-1"])), ordered=True))
            dataset.loc[:,column] = dataset.loc[:,column].fillna("-1")
    #print(cat_feature_ind)
    dataset.to_pickle(f"{data_path}dataset_{months}_{entity}.pkl")

    # Split data into train and test
    # If registry is all, split 80:20, otherwise use the registry as test set
    if registry == "all":
        dataset_train, dataset_test = train_test_split(dataset, test_size=0.2, random_state=21, stratify=dataset[f"liveslongerthan{months}months"])
    else:
        dataset_test = dataset[dataset["Register_ID_FK"] == registry]
        dataset_train = dataset[dataset["Register_ID_FK"] != registry]
    if not args.inverse:
        y_test = dataset_test[f"deathwithin{months}months"]
        y_train = dataset_train[f"deathwithin{months}months"]
    else:
        y_test = dataset_test[f"liveslongerthan{months}months"]
        y_train = dataset_train[f"liveslongerthan{months}months"]

    # Drop columns that are not needed or are leaking information
    drop_cols = ["Register_ID_FK", "Patient_ID_unique", "Diagnosedatum", "Geburtsdatum", "Beginn_Bestrahlung", "Beginn_SYST", "Datum_OP", 
                 "Datum_Vitalstatus", "Verstorben", "survival_time", "Anzahl_Tage_Diagnose_Tod", 
                "Todesursache_Grundleiden", "Todesursache_Grundleiden_Version", f"liveslongerthan{months}months", f"deathwithin{months}months",
                "Anzahl_Tage_Diagnose_ST", "Anzahl_Tage_Diagnose_OP", "Anzahl_Tage_Diagnose_SYST", "Inzidenzort", "Primaertumor_DCN",
                "Anzahl_Tage_SYST", "Anzahl_Tage_ST", "Primaertumor_ICD_Version", "pTNM_Version", "cTNM_Version", "Primaertumor_Morphologie_ICD_O_Version",
                "Primaertumor_Topographie_ICD_O_Version", "Zielgebiet_CodeVersion", "TNM_Version", "Weitere_Todesursachen", "Weitere_Todesursachen_Version",
                "Menge_OPS_version", "Diagnosesicherung", "Protokolle"]
    
    
    obj_cols = dataset.select_dtypes(include=['object']).columns
    drop_cols.extend(obj_cols)


    X_test = dataset_test.drop(columns=drop_cols)
    X_train = dataset_train.drop(columns=drop_cols)
    
    # Get categorical feature indices
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

    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=21)

    # Hyperparameter optimization
    def objective(trial):
        """
        Optimize the parameters for training a CatBoostClassifier model using Optuna.
        Parameters:
        trial (optuna.trial.Trial): The Optuna trial object.
        Returns:
        float: The validation log loss score of the trained model.
        """
        train_pool = Pool(X_train_split, y_train_split, cat_features=cat_feature_ind)
        eval_pool = Pool(X_val, y_val, cat_features=cat_feature_ind)
        param = {
            "objective": "Logloss",
            "eval_metric": "Logloss",
            "early_stopping_rounds": 15,
            "task_type": "CPU",
            "nan_mode": "Min",
            "cat_features": cat_feature_ind,
            "custom_metric": ['Logloss', 'F1', 'Accuracy'],
            "train_dir": f"./catboost_info/{registry}_{months}/",
            "border_count": trial.suggest_int("border_count", 128, 254),
            "iterations": 1000,
            "depth": trial.suggest_int("depth", 3, 10),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 10),
            "random_seed": random_state
        }
        
        model = CatBoostClassifier(**param)
        model.fit(train_pool, eval_set=eval_pool)
        return model.get_best_score()["validation"]["Logloss"]
    
    
    # study = optuna.create_study(study_name=str(datetime.datetime.now()),
    #                             storage=study_db,
    #                             direction="minimize",
    #                             sampler=optuna.samplers.TPESampler(seed=random_state))
    # study.optimize(objective, n_trials=25, n_jobs=4)
    # trial = study.best_trial
    # logger.info(f"Best trial: {trial}")
    # logger.info(f"Best parameters: {trial.params}")
    # model = CatBoostClassifier(**trial.params)

    # Train model
    if args.dummy:
        model = DummyClassifier(strategy="prior")
    else:
    
        model = CatBoostClassifier(early_stopping_rounds=30,
                                task_type="CPU",
                                #devices='cuda:0',
                                nan_mode="Min",
                                cat_features=cat_feature_ind,
                                custom_metric=['Logloss', 'F1', 'Accuracy'],
                                train_dir=f"./catboost_info/{registry}_{months}/",
                                border_count=190,
                                depth=7,
                                colsample_bylevel=0.6,
                                min_data_in_leaf=48,
                                one_hot_max_size=5
                                )
    
    

    train_pool = Pool(X_train_split, y_train_split, cat_features=cat_feature_ind)
    eval_pool = Pool(X_val, y_val, cat_features=cat_feature_ind)
    if args.dummy:
        model.fit(X_train_split, y_train_split)
    else:
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
    plt.savefig(f"./results/{dummy}roc_{registry}_{months}.png", dpi=500)
    plt.close()
    
    # Save predictions
    prediction = model.predict(X_train)
    dataset_train.loc[:,"prediction"] = prediction
    dataset_train[['Patient_ID_unique', 'prediction', f"deathwithin{months}months"]].to_csv(f"./results/{dummy}predictions_{registry}_{months}.csv", index=False)

    # Log results
    logger.info(f"Test Registry: {registry} - binary classification for {months} months")
    logger.info(f"Dataset shape: {dataset.shape}")
    logger.info(f"Train shape: {dataset_train.shape}")
    logger.info(f"Test shape: {dataset_test.shape}")

    logger.info(f"Model score: {model.score(X_test, y_test)}")
    logger.info(metrics.classification_report(y_test, model.predict(X_test)))
    logger.info(f"Area under the ROC curve: {roc_auc}")
    if not args.dummy:
        logger.info(model.get_feature_importance(prettified=True).to_string())
        model.save_model(f"./results/{dummy}model_{registry}_{months}.cbm")
        tree = model.plot_tree(tree_idx=0, pool=train_pool)
        tree.save(f"./results/{dummy}tree_{registry}_{months}.gv")
        os.system(f"dot -Tpng ./results/{dummy}tree_{registry}_{months}.gv -o ./results/tree_{registry}_{months}.png")
    # for i, name in enumerate(X_train.columns):
    #     model.calc_feature_statistics(train_pool, feature=i, plot=True, cat_feature_values=cat_feature_ind ,plot_file=f"./results/features/{name}_{registry}_{months}.png")
