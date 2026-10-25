import pandas as pd
from dotenv import load_dotenv
import os
from pathlib import Path
import shutil
import numpy as np


def get_data(season:int,start_index:int,end_index:int, target_file:str):

    """This function takes in base_dir, season,start_index,end_index (indexes are the number of GW we want to move) and target_file name to move and tag all datasets from our source, that is defined in .env file as DATA_SOURCE_PATH, to our data folder."""
    
    base_dir = Path(__file__).resolve().parent.parent 
    load_dotenv(base_dir / ".env")
    source_path = os.getenv("DATA_SOURCE_PATH")
    if source_path is None:
        raise ValueError("No DATA_SOURCE_PATH set!")

    source_folder = Path(source_path)
    destination_folder = base_dir / "data/raw"
    destination_folder.mkdir(parents=True, exist_ok=True)


    for i in range(start_index,end_index+1):
        file_name = source_folder / f"GW{i}" / f"{target_file}.csv"

        if file_name.is_file():
            new_file = destination_folder / f"{target_file}{season}_{i}.csv"

            shutil.move(str(file_name), str(new_file))

            print(f"succesfully moved from {file_name} to {new_file}")
        else:
            print(f"couldnt move {file_name}")



def create_features(pd_input: pd.DataFrame):
    """Creates all necessary features for prediction model. And saves that dataset in data/processed/features/ ."""

    base_dir = Path(__file__).resolve().parent.parent 
    target_folder = base_dir / "data/processed/features"
    target_folder.mkdir(parents=True, exist_ok=True)


    def shifted_ewm(x,span=5):
        return x.ewm(span=span, adjust=False).mean().shift(1)

    for col in ['team_code', 'home_team', 'away_team']:
        pd_input[col] = pd_input[col].astype(int)

    pd_input['is_home'] = (pd_input['team_code'] == pd_input["home_team"]).astype(int)
    pd_input['opponent_id'] = np.where(pd_input["is_home"] == 0, pd_input["home_team"], pd_input["away_team"])



    position_map = {'Goalkeeper':1,
                    'Defender':2,
                    'Midfielder':3,
                    'Forward':4}

    if not pd_input["position"].isin(position_map).all():
        raise ValueError("Position in input dataframe has unwanted values!")
    pd_input["position_id"] = pd_input['position'].map(position_map).astype('int8')

    pd_input = pd_input.sort_values(by=['player_id','kickoff_time','season','gameweek']).reset_index(drop=True)


    # Rolling EWMA for team xG and xGA
    pd_matches = pd_input.drop_duplicates(subset=['match_id','kickoff_time']).reset_index(drop=True)

    pd_matches_concat = pd.concat([
        pd.DataFrame({
            "gameweek": pd_matches['gameweek'], "team": pd_matches['home_team'],"season": pd_matches["season"],
            "opponent": pd_matches['away_team'], "team_xg": pd_matches['home_expected_goals_xg'],
            "opp_xg": pd_matches['away_expected_goals_xg'], "match_id": pd_matches['match_id'], "kickofftime": pd_matches['kickoff_time']
        }),
        pd.DataFrame({
            "gameweek": pd_matches['gameweek'], "opponent": pd_matches['home_team'],"season": pd_matches["season"],
            "team": pd_matches['away_team'], "opp_xg": pd_matches['home_expected_goals_xg'],
            "team_xg": pd_matches['away_expected_goals_xg'], "match_id": pd_matches['match_id'], "kickofftime": pd_matches['kickoff_time']
            })
    ]).sort_values(['season','gameweek','kickofftime']).reset_index(drop=True)

    team_source = pd_matches_concat.copy()

    pd_matches_concat['xg_for_ewm'] = pd_matches_concat.groupby('team')['team_xg'].transform(
        shifted_ewm
    )
    pd_matches_concat['xg_away_ewm'] = pd_matches_concat.groupby('team')['opp_xg'].transform(
        shifted_ewm
    )

    gw_avg = pd_matches_concat.groupby(['season','gameweek'])['team_xg'].mean()
    league_prior = gw_avg.expanding().mean().shift(1)
    pd_matches_concat['league_xg_avg'] = pd.MultiIndex.from_frame(pd_matches_concat[['season','gameweek']]).map(league_prior)

    pd_matches_concat = pd_matches_concat[["gameweek","team","match_id","xg_for_ewm","xg_away_ewm","league_xg_avg","kickofftime"]]

    team_cols = ["xg_for_ewm","xg_away_ewm"]
    for col in team_cols:
        pd_matches_concat[col] = pd_matches_concat[col].fillna(pd_matches_concat['league_xg_avg'])

    pd_matches_concat["attack_strength"] = pd_matches_concat["xg_for_ewm"] / pd_matches_concat["league_xg_avg"]
    pd_matches_concat["defense_strength"] = pd_matches_concat["xg_away_ewm"] / pd_matches_concat["league_xg_avg"]


    # Merge team stats
    pd_features = pd_input.merge(pd_matches_concat, left_on=["gameweek","match_id","team_code"], 
                                 right_on=["gameweek","match_id","team"])

    columns_rename = ["xg_for_ewm","xg_away_ewm","attack_strength","defense_strength"]

    pd_features = pd_features.drop(columns=["league_xg_avg","team","kickoff_time","kickofftime"])
    pd_features = pd_features.rename(columns={col:f"{col}_team" for col in columns_rename})

    # Merge opponent stats
    pd_features = pd_features.merge(pd_matches_concat, left_on=["gameweek","match_id","opponent_id"], 
                                    right_on=["gameweek","match_id","team"]).drop(columns=["team"])
    
    pd_features = pd_features.rename(columns={col:f"{col}_opp" for col in columns_rename})

    # Create attack and defense indexes between teams
    pd_features["attack_index"] = pd_features["attack_strength_team"]/pd_features["defense_strength_opp"]
    pd_features["defense_index"] = pd_features["attack_strength_opp"]/pd_features["defense_strength_team"] 

    # Create and normalize ewma stats
    per_90_list = ['total_shots'
                    ,'xg','xa'
                    ,'shots_on_target'
                    ,'chances_created'
                    ,'saves','team_goals_conceded'
                    ,'xgot_faced'
                    ,'accurate_crosses'
                    ,'successful_dribbles'
                    ,'big_chances_missed'
                    ,'defensive_contrib']

    

    def_cols = ['clearances', 'blocks', 'interceptions', 'tackles_won', 'recoveries']
    pd_features[def_cols] = pd_features[def_cols].fillna(0)
    
    defensive_cbit = pd_features["clearances"]+pd_features["blocks"]+pd_features["interceptions"]+pd_features["tackles_won"]
    defensive_cbirt = defensive_cbit+pd_features["recoveries"]

    conditions = [
        (pd_features["position_id"] == 2),
        (pd_features["position_id"].isin([3,4]))
    ]
    choices = [defensive_cbit,defensive_cbirt]

    pd_features["defensive_contrib"] = np.select(conditions,choices,default=0)


    team_matches = pd_matches_concat[['match_id', 'gameweek', 'team','kickofftime']]
    span = (pd_features.groupby(['player_id','team_code']).agg(
        first_kick=('kickofftime','min'),last_kick=('kickofftime','max')
    ).reset_index())

    grid = span.merge(team_matches,left_on='team_code', right_on='team')
    grid = grid[((grid['kickofftime'] >= grid['first_kick']) & (grid['kickofftime'] <= grid['last_kick']))]

    stats_cols = per_90_list+["minutes_played"]
    keys = ['player_id','match_id','team_code']
    grid = grid.merge(pd_features[keys+stats_cols],on=keys,how='left')
    grid[stats_cols] = grid[stats_cols].fillna(0)

    grid['is_out'] = (grid['minutes_played'] == 0).astype(int)
    grid['is_partial'] = ((grid['minutes_played'] >=1) & (grid['minutes_played'] <= 59)).astype(int)
    grid['is_full'] = (grid['minutes_played'] >= 60).astype(int)

    grid = grid.sort_values(by=['player_id', 'kickofftime']).reset_index(drop=True)

    grid['minutes_played_ewm'] = grid.groupby('player_id')['minutes_played'].transform(
        shifted_ewm
    )

    merge_cols = ['player_id','match_id','team_code','minutes_played_ewm']

    denom = grid['minutes_played_ewm'].clip(lower=1)

    for stat in per_90_list:
        grid[f'{stat}_ewm'] = grid.groupby('player_id')[stat].transform(shifted_ewm)
        grid[f'{stat}_ewm_per90'] = (grid[f'{stat}'+ '_ewm']/denom)*90
        merge_cols.append(f'{stat}_ewm_per90')
        merge_cols.append(f'{stat}_ewm')

    for bucket in ['is_out','is_partial','is_full']:
        grid[f'p_{bucket}'] = grid.groupby('player_id')[bucket].transform(shifted_ewm)
        merge_cols.append(f'p_{bucket}')

    
    pd_features = pd_features.merge(grid[merge_cols],on=['player_id','match_id','team_code'],how='left')

    # Drop first gameweek
    min_season = pd_features['season'].min()
    min_gw = pd_features.loc[pd_features['season'] == min_season,'gameweek'].min()
    pd_features = pd_features[~((pd_features['season'] == min_season) & (pd_features['gameweek'] == min_gw))]

    # Drop na from playerstats
    player_cols = ['minutes_played_ewm', 'p_is_out', 'p_is_partial', 'p_is_full'] + [c for c in pd_features.columns if c.endswith('_ewm_per90')]
    pd_features =  pd_features.dropna(subset=player_cols).reset_index(drop=True)
    pd_features = pd_features[pd_features['minutes_played'] > 0]
    training_features = ['is_home', 'position_id', 'attack_index', 'defense_index'] + player_cols
    pd_features = pd_features[training_features + ["xg","xa","player_id"]]

    pd_features.to_parquet(target_folder / 'training.parquet', compression='snappy', index=False)


    # Create stat tracking database for players, teams, league, minutes_played

    # Players state

    players_source = grid.sort_values(['player_id','kickofftime']).copy()
    player_ewm_cols = ['minutes_played'] + per_90_list + ['is_out','is_partial','is_full']

    def create_ewm_now(cols_list: list,df_source: pd.DataFrame,group:str | list):
        columns_now = []
        for c in cols_list:
            df_source[f'{c}_now'] = df_source.groupby(group)[c].transform(
                lambda s: s.ewm(span=5, adjust=False).mean() 
            )
            columns_now.append(f'{c}_now')
        return columns_now

    now_cols = create_ewm_now(cols_list=player_ewm_cols,df_source=players_source,group='player_id')

    players_state = (players_source[['player_id','gameweek'] + now_cols]
                     .drop_duplicates(subset='player_id',keep='last')
                     .rename(columns={'gameweek':'last_gw'}))

    players_state.to_parquet(target_folder / 'players_state.parquet', compression='snappy', index=False)


    # Team state

    teams_ewm_cols = create_ewm_now(cols_list=['team_xg', 'opp_xg'],df_source=team_source,group='team')

    teams_state = (team_source[['team','gameweek'] + teams_ewm_cols]
                   .drop_duplicates(subset='team',keep='last')
                   .rename(columns={'gameweek':'last_gw', 'team_xg_now':'xg_for_now', 'opp_xg_now': 'xg_away_now'}))

    teams_state.to_parquet(target_folder / 'teams_state.parquet', compression='snappy', index=False)


    # League average

    gw = team_source.groupby(['season','gameweek'])['team_xg'].mean().reset_index(name='gw_avg')


    league_state = pd.DataFrame({
        'last_gw': [gw['gameweek'].iloc[-1]],
        'league_sum': [gw['gw_avg'].sum()],
        'league_count': [len(gw)]
    })

    league_state.to_parquet(target_folder / 'league_state.parquet', compression='snappy', index=False)



    # Minutes Played

    pd_played = pd_input.loc[pd_input['minutes_played'] > 0, 'minutes_played']

    pd_partial = pd_played[pd_played <= 59]
    pd_full = pd_played[pd_played >= 60]

    partial_bins = [(1, 9), (10, 19), (20, 29), (30, 39), (40, 59)]
    full_bins = [(60, 74), (75, 89), (90, 120)]

    def bin_table(values,buckets):
        rows = [{'low': lo, 'high': hi, 'count': ((values >= lo) & (values <= hi)).sum()} for lo,hi in buckets]

        df = pd.DataFrame(rows)
        df['p'] = df['count'] / df['count'].sum()

        return df

    minutes_partial = bin_table(pd_partial,partial_bins)
    minutes_full = bin_table(pd_full,full_bins)

    minutes_partial.to_parquet(target_folder / 'minutes_partial.parquet', compression='snappy', index=False)
    minutes_full.to_parquet(target_folder / 'minutes_full.parquet', compression='snappy', index=False)

    print("Succesfully created and saved features!")

    


def concat_gameweeks(folder_path:str | Path) -> pd.DataFrame:
    """Reads matching files in folder_path and concatenates them."""
    folder_path = Path(folder_path)

    parquet_files = list(folder_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {folder_path}")
    df_list = []

    for dataset in parquet_files:
        df = pd.read_parquet(dataset, engine='pyarrow')
        df_list.append(df)

    pd_output = pd.concat(df_list, ignore_index=True)

    # Check blank matches
    row_blank = pd_output[['xg', 'xa']].isna().all(axis=1)
    match_blank = row_blank.groupby(pd_output['match_id']).all()  
    pd_output['stats_missing'] = pd_output['match_id'].isin(match_blank[match_blank].index)
    pd_output = pd_output[~pd_output["stats_missing"]]
    pd_output = pd_output.drop(columns=["stats_missing"])
    pd_output = pd_output.fillna(0)
    

    print('Succesfully joined all gameweeks into one DataFrame')
    return pd_output

def join_data(folder_path:str | Path,start_gw:int,end_gw:int, start_season:int,end_season:int):

    if not (1 <= start_gw <= 38 and 1 <= end_gw <= 38):
        raise ValueError("Gameweek must be between 1 and 38!")

    folder_path = Path(folder_path)
    base_dir = Path(__file__).resolve().parent.parent 
    destination = base_dir / "data/processed/merged"
    destination.mkdir(parents=True, exist_ok=True)

    def merge_pandas(season,gameweek):
        pd_playerstats = pd.read_csv(folder_path / f"playermatchstats{season}_{gameweek}.csv")
        pd_matches = pd.read_csv(folder_path / f"matches{season}_{gameweek}.csv")
        pd_players = pd.read_csv(folder_path / f"players{season}_{gameweek}.csv")
        assert pd_players['player_id'].is_unique, f"duplicate player_id in players{season}_{gameweek}.csv"

        pd_stats_matches = pd.merge(left=pd_playerstats,right=pd_matches,on="match_id")
        pd_full = pd.merge(left=pd_stats_matches,right=pd_players,on="player_id")
        pd_full['season'] = season
        pd_full['gameweek'] = gameweek
        pd_full = pd_full.drop(columns="player_id")
        pd_full = pd_full.rename(columns={"player_code":"player_id"})
        print(len(pd_playerstats) == len(pd_full))

        return pd_full

    if start_season == end_season:
        if end_gw < start_gw:
            raise ValueError("start_gw cannot be greater than end_gw within the same season!")
        for week in range(start_gw,end_gw+1):
            pd_merged = merge_pandas(season=start_season,gameweek=week)

            pd_merged.to_parquet(destination / f'merged{start_season}_{week}.parquet', compression='snappy', index=False)


    elif end_season < start_season:
        raise ValueError("start_season cannot be greater than end_season!")
    else:
        for season in range(start_season,end_season+1):

            start_index = start_gw if season==start_season else 1

            end_index = end_gw if season == end_season else 38

            for week in range(start_index,end_index+1):
                pd_merged = merge_pandas(season=season,gameweek=week)

                pd_merged.to_parquet(f'{destination}/merged{season}_{week}.parquet', compression='snappy', index=False)


    print("Succesfully joined datasets")

