import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold

base_dir = Path(__file__).resolve().parent.parent 
destination = base_dir / "data/processed/features/training.parquet"

pd_data = pd.read_parquet(destination, engine='pyarrow')

['is_home', 'position_id', 'attack_index', 'defense_index',
       'minutes_played_ewm', 'p_is_out', 'p_is_partial', 'p_is_full',
       'total_shots_ewm_per90', 'xg_ewm_per90', 'xa_ewm_per90',
       'shots_on_target_ewm_per90', 'chances_created_ewm_per90',
       'tackles_won_ewm_per90', 'interceptions_ewm_per90',
       'recoveries_ewm_per90', 'blocks_ewm_per90', 'clearances_ewm_per90',
       'was_fouled_ewm_per90', 'fouls_committed_ewm_per90', 'saves_ewm_per90',
       'team_goals_conceded_ewm_per90', 'xgot_faced_ewm_per90',
       'penalties_scored_ewm_per90', 'penalties_missed_ewm_per90',
       'accurate_crosses_ewm_per90', 'successful_dribbles_ewm_per90',
       'big_chances_missed_ewm_per90', 'defensive_contrib_ewm_per90', 'xg',
       'xa', 'player_id']

def target_split(pd_input:pd.DataFrame, target:str):
    if target.lower == "xg":
        X = ['is_home', 'attack_index', 'defense_index', 'p_is_out', 'p_is_partial', 'p_is_full',]
    elif target.lower == "xa":
    else:
        raise ValueError("Target has to be xg or xa!")

def tune_parameters(pd_train):
