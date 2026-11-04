import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import requests


def add_prices(players_df: pd.DataFrame):
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    fpl_data = pd.DataFrame(requests.get(url).json()['elements'])
    output_df = players_df.merge(fpl_data[['code','now_cost','can_transact','web_name']], left_on='player_id', right_on='code', how='left')
    output_df = output_df[output_df['can_transact'] == True]
    output_df['now_cost'] = output_df['now_cost']*100_000
    output_df = output_df.drop(columns=['code'])

    return output_df


def best_player(points_df: pd.DataFrame,budget: int) -> pd.DataFrame:

    points_df = points_df.reset_index(drop=True)

    points = points_df['points'].to_numpy()
    prices = points_df['now_cost'].to_numpy()
    position_id = points_df['position_id'].to_numpy()
    players_name = points_df['web_name'].to_numpy()

    gk_mask  = (position_id == 1).astype(float)
    def_mask = (position_id == 2).astype(float)
    mid_mask = (position_id == 3).astype(float)
    fwd_mask = (position_id == 4).astype(float)

    squad_size_constraint = LinearConstraint(A=np.ones(len(players_name)), lb=15, ub=15)

    gk_constraint  = LinearConstraint(A=gk_mask,  lb=2, ub=2)
    def_constraint = LinearConstraint(A=def_mask, lb=5, ub=5)
    mid_constraint = LinearConstraint(A=mid_mask, lb=5, ub=5)
    fwd_constraint = LinearConstraint(A=fwd_mask, lb=3, ub=3)

    budget_constraint = LinearConstraint(A=prices,lb=0,ub=budget)

    bounds = Bounds(lb=0,ub=1)
    integrality = np.ones(len(players_name))
    c = -points

    result = milp(c=c,
                  constraints=[squad_size_constraint, gk_constraint, def_constraint, mid_constraint, fwd_constraint, budget_constraint],
                  integrality=integrality,
                  bounds=bounds)

    if result.success == False:
        print("No feasible squad found:", result.message)
        return
    else:
        chosen = result.x.round().astype(bool)
        squad_names = players_name[chosen]
        squad_cost = prices[chosen].sum()
        squad_points = points[chosen].sum()
        print(f"Squad cost: {squad_cost}, total expected points: {squad_points}")
        return pd.DataFrame({"names": squad_names,"price": prices[chosen]})

    
        
