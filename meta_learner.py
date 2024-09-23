from os import path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_import.data_loading import import_aicare
from data_import.data_preprocessing import calculate_survival_time
import catboost

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC
from ordinal_loss import OrdinalLoss
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from imblearn.over_sampling import RandomOverSampler, SMOTENC
import logging
import datetime

def log_results(logger, model_name, model, X_test, y_test, X_train, y_train):
    logger.info(f"Model: {model_name}")
    if hasattr(model, "score"):
        logger.info(f"Train: {model.score(X_train, y_train)}")
        logger.info(classification_report(y_train, model.predict(X_train), zero_division=0))
        logger.info(f"MAE: {mean_absolute_error(y_train, model.predict(X_train))}")
        logger.info(f"Test: {model.score(X_test, y_test)}")
        logger.info(classification_report(y_test, model.predict(X_test), zero_division=0))
        logger.info(f"MAE: {mean_absolute_error(y_test, model.predict(X_test))}")
    else:
        train_pred, test_pred = model
        logger.info(f"Train: {accuracy_score(y_train, train_pred)}")
        logger.info(classification_report(y_train, train_pred, zero_division=0))
        logger.info(f"MAE: {mean_absolute_error(y_train, train_pred)}")
        logger.info(f"Test: {accuracy_score(y_test, test_pred)}")
        logger.info(classification_report(y_test, test_pred, zero_division=0))
        logger.info(f"MAE: {mean_absolute_error(y_test, test_pred)}")

def simple_decision_rule(predictions, inverse=False):
    """Simple decision rule to convert binary probabilities to multiclass"""
    
    predictions_out = np.zeros([predictions.shape[0], predictions.shape[1]+1])
    if inverse:
        # death within 6 months
        predictions_out[:,0] = predictions[:,0]
        # death within 12 months = death within 12 months - death within 6 months and so on
        for i in range(1, predictions.shape[1]):
            predictions_out[:, i] = predictions[:,i]* (1 - predictions[:,i-1])  
        # last class: death after 24 months
        predictions_out[:,-1] = 1 - predictions[:,-1]
    else:
        # death within 6 months = 1 - lives longer than 6 months
        predictions_out[:,0] = 1 - predictions[:,0]
        # death between 6 and 12 months = lives longer than 6 months - lives longer than 12 months and so on
        for i in range(1, predictions.shape[1]):
            predictions_out[:, i] = predictions[:,i-1] * (1-predictions[:,i])
        # last class: lives longer than 24 months
        predictions_out[:,-1] = predictions[:,-1]
    
    return predictions_out.argmax(axis=1), predictions_out

        


def meta_learner(registry:str, dead_only:bool=False, oversampling:bool=False):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = f"./results/meta_learning_{registry}{"_dead-only" if dead_only else ""}{"_oversampling" if oversampling else ""}.txt"   
    if logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
    logger.addHandler(logging.FileHandler(log_path, mode="w"))
    logger.info(f"Starting meta learning for registry {registry} at the time {datetime.datetime.now()}")
    catboost_models = []
    for month in [6, 12, 18, 24]:
        catboost_models.append(catboost.CatBoostClassifier().load_model(f"./results/model_{registry}_{month}.cbm"))
    # Load data
    if path.exists(f"./results/dataset_metalearner{"_dead-only" if dead_only else ""}.pkl"):
        dataset = pd.read_pickle(f"./results/dataset_metalearner{"_dead-only" if dead_only else ""}.pkl")
    else:
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

        dataset = dataset[dataset["survival_time"]>0]
        if dead_only:
            dataset = dataset[dataset["Verstorben"] == 1]

        dataset["Alter_bei_Diagnose"] = ((dataset['Diagnosedatum'] - dataset['Geburtsdatum']).dt.days // 365.25).astype(int)


        dataset["survival_class"] = 0
        for month in [6, 12, 18, 24]:
            dataset.loc[:,"survival_class"][dataset["survival_time"] > 30 * month] = dataset.loc[:,"survival_class"] + 1

        for column in dataset.select_dtypes(include=['category']).columns:
            dataset.loc[:,column] = dataset.loc[:,column].astype(pd.CategoricalDtype(categories=dataset[column].cat.categories.append(pd.Index(["-1"])), ordered=True))
            dataset.loc[:,column] = dataset.loc[:,column].fillna("-1")

        dataset.to_pickle(f"./results/dataset_metalearner{"_dead-only" if dead_only else ""}.pkl")
    #print(cat_feature_ind)
    if registry == "all":
        dataset_train, dataset_test = train_test_split(dataset, test_size=0.2)
    else:
        dataset_test = dataset[dataset["Register_ID_FK"] == registry]
        dataset_train = dataset[dataset["Register_ID_FK"] != registry]
    y_test = dataset_test["survival_class"].to_numpy()

    drop_cols = ["Register_ID_FK", "Patient_ID_unique", "Diagnosedatum", "Geburtsdatum", "Beginn_Bestrahlung", "Beginn_SYST", "Datum_OP", 
                 "Datum_Vitalstatus", "Verstorben", "survival_time", "Anzahl_Tage_Diagnose_Tod", 
                "Todesursache_Grundleiden", "Todesursache_Grundleiden_Version", "survival_class",
                "Anzahl_Tage_Diagnose_ST", "Anzahl_Tage_Diagnose_OP", "Anzahl_Tage_Diagnose_SYST", "Inzidenzort", "Primaertumor_DCN",
                "Anzahl_Tage_SYST", "Anzahl_Tage_ST", "Primaertumor_ICD_Version", "pTNM_Version", "cTNM_Version", "Primaertumor_Morphologie_ICD_O_Version",
                "Primaertumor_Topographie_ICD_O_Version", "Zielgebiet_CodeVersion", "TNM_Version", "Weitere_Todesursachen", "Weitere_Todesursachen_Version",
                "Menge_OPS_version", "Diagnosesicherung", "Protokolle"]
    
    obj_cols = dataset.select_dtypes(include=['object']).columns
    drop_cols.extend(obj_cols)


    X_test = dataset_test.drop(columns=drop_cols)
    y_train = dataset_train["survival_class"].to_numpy()
    print(y_train.shape)
    X_train = dataset_train.drop(columns=drop_cols)
        

    cat_feature_ind = [X_train.columns.get_loc(c) for c in X_train.columns if X_train[c].dtype.name == 'category']
    text_feature_ind = [X_train.columns.get_loc(c) for c in X_train.columns if X_train[c].dtype.name == 'string']
    for column in X_train.select_dtypes(include=['string']).columns:
        X_train.loc[:,column] = X_train.loc[:,column].fillna("-1")
        X_test.loc[:,column] = X_test.loc[:,column].fillna("-1")
    for column in X_train.select_dtypes(include=['int']).columns:
        X_train.loc[:,column] = X_train.loc[:,column].fillna(-1)
        X_test.loc[:,column] = X_test.loc[:,column].fillna(-1)
    cat_feature_ind.extend(text_feature_ind)
    #print number of nan values
    pd.set_option('display.max_rows', None)
    print(X_train.isna().sum())
    #  if oversampling:
        #implement sqrt balanced class weights
        #X_train, y_train = RandomOverSampler(random_state=21).fit_resample(X_train, y_train)
        # categorical_features=cat_feature_ind,
    # else:
    #    class_weights = None

    print("Logistic Regression")
    stack = StackingClassifier(estimators=[("catboost_6", catboost_models[0]), ("catboost_12", catboost_models[1]), ("catboost_18", catboost_models[2]), ("catboost_24", catboost_models[3])],
                               final_estimator=LogisticRegressionCV(cv=5, max_iter=1000, n_jobs=-1, class_weight=("balanced" if oversampling else None), verbose=1, random_state=21),
                               cv="prefit"
                )
    
    stack.fit(X_train, y_train)
    log_results(logger, "Stack with Logistic Regression", stack, X_test, y_test, X_train, y_train)
    
    print("CatBoost")
    stack = StackingClassifier(estimators=[("catboost_6", catboost_models[0]), ("catboost_12", catboost_models[1]), ("catboost_18", catboost_models[2]), ("catboost_24", catboost_models[3])],
                               final_estimator=catboost.CatBoostClassifier(early_stopping_rounds=30,
                                                                           task_type="CPU",
                                                                           nan_mode="Min",
                                                                           #cat_features=cat_feature_ind,
                                                                           loss_function="MultiClass",
                                                                           auto_class_weights=("Balanced" if oversampling else None),
                                                                           verbose=1,
                                                                           random_seed=21),
                               cv="prefit",
                               passthrough=False
                )
    
    stack.fit(X_train, y_train)
    log_results(logger, "Stack with CatBoost", stack, X_test, y_test, X_train, y_train)

    print("SVC")
    stack = StackingClassifier(estimators=[("catboost_6", catboost_models[0]), ("catboost_12", catboost_models[1]), ("catboost_18", catboost_models[2]), ("catboost_24", catboost_models[3])],
                               final_estimator=LinearSVC(verbose=0,
                                                         class_weight=("balanced" if oversampling else None),
                                                         random_state=21),
                               cv="prefit"
                )
    
    stack.fit(X_train, y_train)
    log_results(logger, "Stack with SVC", stack, X_test, y_test, X_train, y_train)
    


    X_train = catboost.Pool(X_train, cat_features=cat_feature_ind, label=y_train)
    intermediate_predictions = []
    for model in catboost_models:
        
        pred = model.predict(X_train, prediction_type='Probability')[:, 1]
        intermediate_predictions.append(pred)
    intermediate_predictions = np.stack(intermediate_predictions).T
    final_predictions_train, proba = simple_decision_rule(intermediate_predictions)


    print("Simple decision rule")
    intermediate_predictions = []
    for model in catboost_models:
        pred = model.predict(catboost.Pool(X_test, cat_features=cat_feature_ind), prediction_type='Probability')[:, 1]
        intermediate_predictions.append(pred)
    intermediate_predictions = np.stack(intermediate_predictions).T
    final_predictions_test, proba = simple_decision_rule(intermediate_predictions)

    log_results(logger, "Simple decision rule", (final_predictions_train, final_predictions_test), X_test, y_test, X_train, y_train)

    final_cat = catboost.CatBoostClassifier(early_stopping_rounds=20,
                               task_type="CPU",
                               nan_mode="Min",
                               cat_features=cat_feature_ind,
                               loss_function="MultiClass", #OrdinalLoss(),
                               custom_metric=['F1', "Accuracy"],
                               auto_class_weights=("Balanced" if oversampling else None),
                               train_dir=f"./catboost_info/{registry}_multiclass/",
                               verbose=0
                               )
    final_cat.fit(X_train)
    logger.info(f"Model: CatBoost with multiclass")
    logger.info(f"Train: {final_cat.score(X_train)}")
    logger.info(classification_report(y_train, final_cat.predict(X_train), zero_division=0))
    logger.info(f"MAE: {mean_absolute_error(y_train, final_cat.predict(X_train))}")
    logger.info(f"Test: {final_cat.score(X_test, y_test)}")
    logger.info(classification_report(y_test, final_cat.predict(X_test), zero_division=0))
    logger.info(f"MAE: {mean_absolute_error(y_test, final_cat.predict(X_test))}")
if __name__ == "__main__":
    dead_only = True
    oversampling = True
    meta_learner("all", dead_only=dead_only, oversampling=oversampling)
    # meta_learner("1", dead_only=dead_only, oversampling=oversampling)
    # meta_learner("2", dead_only=dead_only, oversampling=oversampling)
    # meta_learner("3", dead_only=dead_only, oversampling=oversampling)
    # meta_learner("5", dead_only=dead_only, oversampling=oversampling)
    # meta_learner("10", dead_only=dead_only, oversampling=oversampling)
    # meta_learner("14", dead_only=dead_only, oversampling=oversampling)
    # meta_learner("15", dead_only=dead_only, oversampling=oversampling)


