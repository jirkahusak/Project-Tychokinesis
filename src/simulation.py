import numpy as np
import pandas as pd
from pathlib import Path

def monte_carlo(xg_grid: pd.DataFrame,xa_grid: pd.DataFrame,saves_grid: pd.DataFrame,other_features: pd.DataFrame, n_runs: int) -> pd.DataFrame:

    base_dir = Path(__file__).resolve().parent.parent 
    destination = base_dir / "data/processed/features"
    pd_partial = pd.read_parquet(destination / "minutes_partial.parquet", engine='pyarrow')
    pd_full = pd.read_parquet(destination / "minutes_full.parquet", engine='pyarrow')

    rng = np.random.default_rng()

    other_features = other_features.reset_index(drop=True)
    player_ids = other_features['player_id'].to_numpy()
    players_count = len(other_features)

    xa_grid = xa_grid.reindex(player_ids, fill_value=0)
    xg_grid = xg_grid.reindex(player_ids, fill_value=0)
    saves_grid = saves_grid.reindex(player_ids, fill_value=0)

    # Choose random playtime bucket
    probs = other_features[['is_out','is_partial', 'is_full']].to_numpy()
    sums = np.cumsum(probs, axis=1)

    threshold0 = sums[:,[0]]
    threshold1 = sums[:,[1]]

    random_bucket = rng.random((players_count,n_runs))
    bucket_idx = (random_bucket >= threshold0).astype(int) + (random_bucket >= threshold1).astype(int)

    time_points = np.select([bucket_idx == 0,bucket_idx==1,bucket_idx==2],
                            [0,1,2])

    # Choose minutes
    partial_bin_idx = rng.choice(len(pd_partial),size=(players_count,n_runs), p=pd_partial['p'].to_numpy())
    low_partial = pd_partial['low'].to_numpy()[partial_bin_idx]
    high_partial = pd_partial['high'].to_numpy()[partial_bin_idx]
    minutes_if_partial = rng.integers(low_partial,high_partial+1)

    full_bin_idx = rng.choice(len(pd_full),size=(players_count,n_runs), p=pd_full['p'].to_numpy())
    low_full = pd_full['low'].to_numpy()[full_bin_idx]
    high_full = pd_full['high'].to_numpy()[full_bin_idx]
    minutes_if_full = rng.integers(low_full,high_full+1)

    minutes = np.select([bucket_idx == 0,bucket_idx==1,bucket_idx==2],
                        [0, minutes_if_partial, minutes_if_full])

    # Choose defensive contrib
    def_contrib = other_features['defensive_contrib'].to_numpy()[:,None]
    lambda_def = def_contrib*(minutes/90)
    is_goalie = (other_features['position_id'].to_numpy() == 1)[:,None]
    lambda_def = np.where(is_goalie,0,lambda_def)

    def_draws = rng.poisson(lambda_def)

    position_id = other_features['position_id'].to_numpy()[:,None]
    threshold_def = np.select([position_id == 2, np.isin(position_id, [3,4])],
                              [10, 12], default=np.inf)

    def_contrib_points = np.where(def_draws>=threshold_def,2,0)

    # Choose xg,xa and saves.
    player_idx = np.arange(players_count)[:,None]

    xa_grid = xa_grid.to_numpy()
    xg_grid = xg_grid.to_numpy()
    saves_grid = saves_grid.to_numpy()

    minutes_for_lookup = np.clip(minutes, 1, 90)
    xg_lookup = xg_grid[player_idx, minutes_for_lookup - 1]
    xa_lookup = xa_grid[player_idx, minutes_for_lookup - 1]
    saves_lookup = saves_grid[player_idx, minutes_for_lookup - 1]

    xg_draws = rng.poisson(xg_lookup)
    xg_draws = np.where(minutes == 0, 0, xg_draws)
    goal_points = np.select([position_id==1,position_id==2,position_id==3,position_id==4],
                            [0,xg_draws*6,xg_draws*5,xg_draws*4])

    xa_draws = rng.poisson(xa_lookup)
    xa_draws = np.where(minutes == 0, 0, xa_draws)
    assist_points = xa_draws*3

    saves_draws = rng.poisson(saves_lookup)
    saves_draws = np.where(minutes == 0, 0, saves_draws)
    saves_points = np.where(position_id==1,saves_draws//3,0)

    # Clean sheet
    goals_df = pd.DataFrame(xg_draws,index=other_features['team'].to_numpy())
    team_goals = goals_df.groupby(level=0).sum()
    opponent_team_goals = team_goals.reindex(other_features['opponent'].to_numpy()).to_numpy()
    clean_sheet = (opponent_team_goals==0) & (minutes >= 60)

    clean_sheet_points = np.select([np.isin(position_id,[1,2]), position_id==3],
                                   [clean_sheet*4,clean_sheet], default=0)

    # Conceded minus points
    conceded_points = np.where(np.isin(position_id,[1,2]) & (minutes > 0), (opponent_team_goals//2)*-1,0)

    total_points = (time_points+def_contrib_points+goal_points+assist_points+saves_points+clean_sheet_points+conceded_points)

    avg_points = total_points.mean(axis=1)

    output_pd = pd.DataFrame({
        "player_id": player_ids,
        "position_id": other_features['position_id'].to_numpy(),
        "points": avg_points
    })

    return output_pd
