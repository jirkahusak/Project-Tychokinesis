import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor

def map_id(players_df):
    position_map = {'Goalkeeper':1,
                    'Defender':2,
                    'Midfielder':3,
                    'Forward':4}
    
    if not players_df["position"].isin(position_map).all():
        raise ValueError("Position in input dataframe has unwanted values!")
    positions = players_df['position'].map(position_map).astype('int8')

    return positions


def concat_matches(pd_matches):
    pd_concat = pd.concat([
                        pd.DataFrame({
                            "gameweek": pd_matches['gameweek'], "team": pd_matches['home_team'],"season": pd_matches["season"],
                            "opponent": pd_matches['away_team'], "team_xg": pd_matches['home_expected_goals_xg'],
                            "opp_xg": pd_matches['away_expected_goals_xg'], "match_id": pd_matches['match_id'], "kickofftime": pd_matches['kickoff_time'],
                            "is_home": 1
                        }),
                        pd.DataFrame({
                            "gameweek": pd_matches['gameweek'], "opponent": pd_matches['home_team'],"season": pd_matches["season"],
                            "team": pd_matches['away_team'], "opp_xg": pd_matches['home_expected_goals_xg'],
                            "team_xg": pd_matches['away_expected_goals_xg'], "match_id": pd_matches['match_id'], "kickofftime": pd_matches['kickoff_time'],
                            "is_home": 0
                            })
                    ]).sort_values(['season','gameweek','kickofftime']).reset_index(drop=True)
    return pd_concat


def update_state(start_season: int, end_season: int, end_gw: int):
    """Updates state database with new data"""

    
    if end_gw > 38:
        raise ValueError("Gameweek cannot be greater than 38!")
    if start_season > end_season:
        raise ValueError("start_season cannot be greater than end_season")

    base_dir = Path(__file__).resolve().parent.parent 
    destination = base_dir / "data/processed/features"

    pd_league = pd.read_parquet(destination / "league_state.parquet", engine='pyarrow')
    pd_players = pd.read_parquet(destination / "players_state.parquet", engine='pyarrow')
    pd_teams = pd.read_parquet(destination / "teams_state.parquet", engine='pyarrow')


    for season in range(start_season,end_season+1):
        start_gw = int(pd_league['last_gw'].iloc[0])+1 if season == start_season else 1
        end_gw_season = end_gw if season == end_season else 38
        for gw in range(start_gw, end_gw_season+1):

            playermatchstats_url = f"https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/refs/heads/main/data/20{season}-20{season+1}/By%20Tournament/Premier%20League/GW{gw}/playermatchstats.csv"
            matches_url =          f"https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/refs/heads/main/data/20{season}-20{season+1}/By%20Tournament/Premier%20League/GW{gw}/matches.csv"
            players_url =          f"https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/refs/heads/main/data/20{season}-20{season+1}/By%20Tournament/Premier%20League/GW{gw}/players.csv"


            pd_playermatchstats_new = pd.read_csv(playermatchstats_url)
            pd_matches = pd.read_csv(matches_url)
            pd_matches["kickoff_time"] = pd.to_datetime(pd_matches["kickoff_time"], format="ISO8601", utc=True)
            pd_players_new = pd.read_csv(players_url)

            row_blank = pd_playermatchstats_new[['xg', 'xa']].isna().all(axis=1)
            match_blank = row_blank.groupby(pd_playermatchstats_new['match_id']).all()
            match_idx = match_blank[match_blank].index

            pd_playermatchstats_new = pd_playermatchstats_new[~pd_playermatchstats_new['match_id'].isin(match_idx)]
            pd_matches = pd_matches[~pd_matches['match_id'].isin(match_idx)]

            position_map = {'Goalkeeper':1,
                            'Defender':2,
                            'Midfielder':3,
                            'Forward':4}

            if not pd_players_new["position"].isin(position_map).all():
                raise ValueError("Position in input dataframe has unwanted values!")
            pd_players_new["position_id"] = pd_players_new['position'].map(position_map).astype('int8')

            pd_matches_concat = concat_matches(pd_matches=pd_matches)

            player_cols = ['player_id','minutes_played', 'total_shots','xg','xa','shots_on_target',
                           'chances_created','saves','team_goals_conceded','xgot_faced',
                           'accurate_crosses','successful_dribbles','big_chances_missed','clearances',
                            'blocks', 'interceptions', 'tackles_won', 'recoveries']


            pd_grid = pd_matches_concat.merge(pd_players_new[["player_id", "player_code", "team_code","position_id"]],left_on="team", right_on="team_code")
            pd_grid = pd_grid.merge(pd_playermatchstats_new[player_cols+["match_id"]],on=["player_id", "match_id"], how='left')
            pd_grid[player_cols] = pd_grid[player_cols].fillna(0)


            pd_grid['is_out'] = (pd_grid['minutes_played'] == 0).astype(int)
            pd_grid['is_partial'] = ((pd_grid['minutes_played'] >=1) & (pd_grid['minutes_played'] <= 59)).astype(int)
            pd_grid['is_full'] = (pd_grid['minutes_played'] >= 60).astype(int)

            conditions = [
                    (pd_grid["position_id"] == 2),
                    (pd_grid["position_id"].isin([3,4]))
                ]

            defensive_cbit = pd_grid["clearances"]+pd_grid["blocks"]+pd_grid["interceptions"]+pd_grid["tackles_won"]
            defensive_cbirt = defensive_cbit+pd_grid["recoveries"]
            choices = [defensive_cbit,defensive_cbirt]

            pd_grid["defensive_contrib"] = np.select(conditions,choices,default=0)


            pd_grid = pd_grid.drop(columns="player_id")
            pd_grid = pd_grid.rename(columns={"player_code":"player_id"})

            create_ewm = lambda series: series.ewm(span=5, adjust=False).mean().iloc[-1]

            def update_statistic(state_df,state_name,name_value,state_col_name,statistics_df,statistics_col_name):
                last_stat = state_df[state_df[state_name] == name_value][state_col_name].iloc[0]
                stat_series = pd.concat([pd.Series([last_stat]), statistics_df[statistics_col_name]],ignore_index=True)
                state_df.loc[state_df[state_name] == name_value, state_col_name] = create_ewm(stat_series)


            # Update team state
            for team,stats in pd_matches_concat.sort_values('kickofftime').groupby('team'):
                if team in pd_teams['team'].values:
                    update_statistic(pd_teams,'team',team,'xg_for_now', stats,'team_xg')
                    update_statistic(pd_teams,'team',team, 'xg_away_now', stats, 'opp_xg')

                    pd_teams.loc[pd_teams['team'] == team, 'last_gw'] = stats['gameweek'].iloc[-1]
                else:
                    # TODO: Look if you can use our function!
                    new_team = pd.DataFrame({'team':[team],'last_gw': [stats['gameweek'].iloc[-1]],'xg_for_now':[create_ewm(stats['team_xg'])],'xg_away_now':[create_ewm(stats['opp_xg'])]})
                    pd_teams = pd.concat([pd_teams,new_team], ignore_index=True)


            # Update league state
            gw_avg = pd_matches_concat['team_xg'].mean()
            pd_league["last_gw"] = pd_matches_concat['gameweek'].max()
            pd_league['league_sum'] = pd_league['league_sum']+gw_avg
            pd_league["league_count"] = pd_league['league_count']+1


            # Update player state 
            player_cols_now = ['total_shots'
                              ,'xg','xa'
                              ,'shots_on_target'
                              ,'chances_created'
                              ,'saves','team_goals_conceded'
                              ,'xgot_faced'
                              ,'accurate_crosses'
                              ,'successful_dribbles'
                              ,'big_chances_missed'
                              ,'defensive_contrib'
                              ,'minutes_played'
                              ,'is_out','is_partial'
                              ,'is_full']

            for player,player_stats in pd_grid.sort_values('kickofftime').groupby('player_id'):
                if player in pd_players['player_id'].values:
                    for col in player_cols_now:
                        update_statistic(pd_players, 'player_id', player, f'{col}_now',player_stats, col)

                    pd_players.loc[pd_players['player_id'] == player, 'last_gw'] = player_stats['gameweek'].iloc[-1]

                else:
                    facts_dict = {'player_id':[player], 'last_gw': player_stats['gameweek'].iloc[-1]}
                    stats_dict = {}
                    for col in player_cols_now:
                        stats_dict[f'{col}_now'] = create_ewm(player_stats[col])

                    new_player = pd.DataFrame(facts_dict | stats_dict)
                    pd_players = pd.concat([pd_players, new_player], ignore_index=True)

            # TODO: update minutes states

            pd_league.to_parquet(destination / "league_state.parquet", compression='snappy', index=False)
            pd_players.to_parquet(destination / "players_state.parquet", compression='snappy', index=False)
            pd_teams.to_parquet(destination / "teams_state.parquet", compression='snappy', index=False)
            print(f"Updated states for Season {season} and Gameweek {gw}.")



def scrape_matches(season: int, gameweek: int):
    """Returns all teams and players that could be playing in a gameweek"""
    matches_url = f"https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/refs/heads/main/data/20{season}-20{season+1}/By%20Tournament/Premier%20League/GW{gameweek}/matches.csv"
    players_url = f"https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/refs/heads/main/data/20{season}-20{season+1}/By%20Tournament/Premier%20League/GW{gameweek}/players.csv"

    pd_matches = pd.read_csv(matches_url)
    pd_matches["kickoff_time"] = pd.to_datetime(pd_matches["kickoff_time"], format="ISO8601", utc=True)
    pd_players = pd.read_csv(players_url)
    pd_players["position_id"] = map_id(pd_players)

    pd_teams = concat_matches(pd_matches=pd_matches)

    pd_gameweek = pd_teams.merge(pd_players[["player_id", "player_code", "team_code","position_id"]], left_on='team', right_on='team_code')

    pd_gameweek = pd_gameweek.drop(columns="player_id")
    pd_gameweek = pd_gameweek.rename(columns={"player_code":"player_id"})


    return pd_gameweek[['player_id','team','opponent','is_home','position_id','kickofftime','gameweek']]


def prepare_features(matches_pd: pd.DataFrame) -> pd.DataFrame:
    """Prepares features for prediction based on position (1/Goalkeeper or 0/None)"""
    base_dir = Path(__file__).resolve().parent.parent 
    destination = base_dir / "data/processed/features"

    pd_league = pd.read_parquet(destination / "league_state.parquet", engine='pyarrow')
    pd_players = pd.read_parquet(destination / "players_state.parquet", engine='pyarrow')
    pd_teams = pd.read_parquet(destination / "teams_state.parquet", engine='pyarrow')


    league_avg = pd_league["league_sum"].iloc[0] / pd_league['league_count'].iloc[0]

    matches_pd = matches_pd.merge(pd_teams[['team', 'xg_for_now', 'xg_away_now']], on='team', how='left')
    matches_pd['attack_strength_team'] = matches_pd['xg_for_now'] / league_avg
    matches_pd['defense_strength_team'] = matches_pd['xg_away_now'] / league_avg
    matches_pd = matches_pd.drop(['xg_for_now', 'xg_away_now'], axis=1)

    matches_pd = matches_pd.merge(pd_teams[['team', 'xg_for_now', 'xg_away_now']], left_on='opponent', right_on='team', how='left', suffixes=('','_right')).drop(columns=['team_right'])
    matches_pd['attack_strength_opp'] = matches_pd['xg_for_now'] / league_avg
    matches_pd['defense_strength_opp'] = matches_pd['xg_away_now'] / league_avg
    matches_pd = matches_pd.drop(['xg_for_now', 'xg_away_now'], axis=1)

    matches_pd['attack_index'] = matches_pd['attack_strength_team'] / matches_pd['defense_strength_opp']
    matches_pd['defense_index'] = matches_pd['attack_strength_opp'] / matches_pd['defense_strength_team']


    matches_pd = matches_pd.merge(pd_players[['player_id', 'minutes_played_now']], on='player_id', how='left')
    matches_pd = matches_pd.rename(columns={"minutes_played_now": 'minutes_played'})


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

    denom =  matches_pd['minutes_played'].clip(lower=1)
    for stat in per_90_list:
        matches_pd = matches_pd.merge(pd_players[['player_id',f'{stat}_now']], on='player_id', how='left')
        matches_pd[f'{stat}_ewm_per90'] = (matches_pd[f'{stat}_now'] / denom)*90
        matches_pd = matches_pd.drop(columns=[f'{stat}_now'])

    for bucket in ['is_out','is_partial','is_full']:
        matches_pd = matches_pd.merge(pd_players[['player_id',f'{bucket}_now']], on='player_id', how='left')
        matches_pd = matches_pd.rename(columns={f'{bucket}_now': bucket})
        


    xa_features = pd.DataFrame({"is_home": matches_pd['is_home'], "attack_index": matches_pd['attack_index'], "defense_index": matches_pd['defense_index'],
                                "xa_ewm_per90": matches_pd['xa_ewm_per90'], "position_id": matches_pd['position_id'], "chances_created_ewm_per90": matches_pd['chances_created_ewm_per90'],
                                "accurate_crosses_ewm_per90": matches_pd['accurate_crosses_ewm_per90'], "successful_dribbles_ewm_per90": matches_pd['successful_dribbles_ewm_per90'], "player_id": matches_pd['player_id']})

    xg_features = pd.DataFrame({"is_home": matches_pd['is_home'], "attack_index": matches_pd['attack_index'], "defense_index": matches_pd['defense_index'],
                                "total_shots_ewm_per90": matches_pd['total_shots_ewm_per90'], "xg_ewm_per90": matches_pd['xg_ewm_per90'], "shots_on_target_ewm_per90": matches_pd['shots_on_target_ewm_per90'],
                                "successful_dribbles_ewm_per90": matches_pd['successful_dribbles_ewm_per90'], "big_chances_missed_ewm_per90": matches_pd['big_chances_missed_ewm_per90'],
                                "position_id": matches_pd['position_id'], "player_id": matches_pd['player_id']})
    
    saves_features = pd.DataFrame({"is_home": matches_pd['is_home'], "attack_index": matches_pd['attack_index'], "defense_index": matches_pd['defense_index'],
                                    "saves_ewm_per90": matches_pd['saves_ewm_per90'], "xgot_faced_ewm_per90": matches_pd['xgot_faced_ewm_per90'], "position_id": matches_pd['position_id'], "player_id": matches_pd['player_id']})
    
    other_features = pd.DataFrame({"is_out": matches_pd['is_out'], "is_partial": matches_pd['is_partial'], "is_full": matches_pd['is_full'], "position_id": matches_pd['position_id'], "defensive_contrib": matches_pd['defensive_contrib'], 
                                   "player_id": matches_pd['player_id'], "team": matches_pd['team'], "opponent": matches_pd['opponent']})
    
    xg_features = xg_features[xg_features['position_id'] != 1]
    xa_features = xa_features[xa_features['position_id'] != 1]
    saves_features = saves_features[saves_features['position_id'] == 1]
    saves_features = saves_features.drop(columns=['position_id'])

    return xg_features,xa_features,other_features, saves_features
    

def predict_player(model: XGBRegressor, features_pd: pd.DataFrame) -> pd.DataFrame:
    minutes_grid = np.arange(1,91)
    minutes_df = pd.DataFrame({"minutes_played": minutes_grid})
    prediction_pd = features_pd.copy()
    batch = prediction_pd.merge(minutes_df, how='cross')
    lambda_values = model.predict(batch.drop(columns='player_id'))

    batch['lambda'] = lambda_values

    return batch.pivot(index='player_id', columns='minutes_played',values='lambda')