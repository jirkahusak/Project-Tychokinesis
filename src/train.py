import pandas as pd
import numpy as np
import optuna
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_poisson_deviance, r2_score
import joblib


def target_split(pd_input:pd.DataFrame, target:str) -> tuple[pd.DataFrame, pd.Series]:
    """Creates X and y for pd_input based on target (xg, xa or saves)."""


    if target.lower() == "xg":
        pd_input = pd_input[pd_input["position_id"] != 1]
        X = pd_input[['is_home', 'attack_index', 'defense_index','total_shots_ewm_per90', 'xg_ewm_per90',
             'shots_on_target_ewm_per90', 'successful_dribbles_ewm_per90', 'big_chances_missed_ewm_per90'
             ,'position_id','minutes_played']]
        y = pd_input["xg"]
    elif target.lower() == "xa":
        pd_input = pd_input[pd_input["position_id"] != 1]
        X = pd_input[['is_home', 'attack_index', 'defense_index', 'xa_ewm_per90', 'position_id',
             'chances_created_ewm_per90', 'accurate_crosses_ewm_per90', 'successful_dribbles_ewm_per90','minutes_played']]
        y = pd_input["xa"]
    elif target.lower() == "saves":
        pd_input = pd_input[pd_input["position_id"] == 1]
        X = pd_input[['is_home', 'attack_index', 'defense_index', 'saves_ewm_per90', 'xgot_faced_ewm_per90','minutes_played']]
        y = pd_input['saves']
    else:
        raise ValueError("Target has to be xg, xa or saves!")

    assert not y.isna().any()
    return X,y


def tune_parameters(pd_train: pd.DataFrame,target_val: str):
    """Tunes hyperparameters of XGBoost model with optuna."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    pd_train = pd_train.sort_values('kickofftime').reset_index(drop=True)
    X_full,y_full = target_split(pd_train,target=target_val)


    def objective(trial: optuna.trial.Trial):

        params = {
            "objective": "count:poisson",
            "eval_metric": "poisson-nloglik",
            "n_estimators": 500,
            "max_delta_step": trial.suggest_float('max_delta_step',low=0.3,high=0.9),
            "max_depth": trial.suggest_int('max_depth',low=3,high=8),
            "min_child_weight": trial.suggest_int('min_child_weight',low=1,high=8),
            "learning_rate": trial.suggest_float('learning_rate',low=0.01,high=0.03,log=True),
            "subsample": trial.suggest_float('subsample',low=0.5,high=1),
            "colsample_bytree": trial.suggest_float('colsample_bytree',low=0.5,high=1),
            "alpha": trial.suggest_float('alpha',low=1e-8,high=8,log=True),
            "lambda": trial.suggest_float('lambda',low=1e-8,high=8,log=True),
            "n_jobs": -1
        }

        tsf = TimeSeriesSplit(n_splits=5)
        errors = []
        r2 = []

        for train_index,val_index in tsf.split(X_full,y_full):
            X_trn,y_trn = X_full.iloc[train_index],y_full.iloc[train_index]
            X_val,y_val = X_full.iloc[val_index],y_full.iloc[val_index]

            model = XGBRegressor(**params)
            model.fit(X_trn,y_trn)

            predictions = model.predict(X_val)


            errors.append(mean_poisson_deviance(y_true=y_val,y_pred=predictions))
            r2.append(r2_score(y_true=y_val,y_pred=predictions))

        trial.set_user_attr("r2_score",float(np.mean(r2)))
        return np.mean(errors)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective,n_trials=50)

    print(f"Best Poisson deviance: {study.best_value}")
    best_trial = study.best_trial
    print(f"Best R2 score: {best_trial.user_attrs['r2_score']}")

    return study.best_params

def train_model(pd_data: pd.DataFrame, best_params: dict, target: str):
    """Trains model for specific target."""
    base_dir = Path(__file__).resolve().parent.parent 
    destination = base_dir / "data/models"
    destination.mkdir(parents=True, exist_ok=True)
    file_path = destination / f"{target}_model.joblib"


    pd_data = pd_data.sort_values('kickofftime').reset_index(drop=True)
    X_full,y_full = target_split(pd_input=pd_data,target=target)

    params = {**best_params, "n_jobs": -1, "n_estimators": 500, "objective": "count:poisson", "eval_metric": "poisson-nloglik",}
    model = XGBRegressor(**params)
    model.fit(X_full,y_full)

    joblib.dump(model,file_path)

def main():
    base_dir = Path(__file__).resolve().parent.parent 
    destination = base_dir / "data/processed/features/training.parquet"

    pd_data = pd.read_parquet(destination, engine='pyarrow')

    for target in ['xg','xa', 'saves']:
        parameters = tune_parameters(pd_train=pd_data,target_val=target)
        train_model(pd_data=pd_data,best_params=parameters,target=target)

        print(f"Successfully trained model for {target}!")

if __name__ == "__main__":
    main()



            
