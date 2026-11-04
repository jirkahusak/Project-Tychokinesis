import joblib
from pathlib import Path

from predict import scrape_matches,prepare_features,predict_player
from simulation import monte_carlo
from optimize import add_prices,best_player


def main(season:int, gameweek:int, budget:int = 100_000_000, n_runs:int = 50_000):
    base_dir = Path(__file__).resolve().parent.parent 
    models_dir = base_dir / "data/models"

    # Load models
    xg_model = joblib.load(models_dir / "xg_model.joblib")
    xa_model = joblib.load(models_dir / "xa_model.joblib")
    saves_model = joblib.load(models_dir / "saves_model.joblib")

    # Setup
    matches_pd = scrape_matches(season=season,gameweek=gameweek)
    xg_features,xa_features,other_features, saves_features = prepare_features(matches_pd=matches_pd)

    xg_grid = predict_player(xg_model, xg_features)
    xa_grid = predict_player(xa_model, xa_features)
    saves_grid = predict_player(saves_model, saves_features)

    # Simulation
    simulated_pd = monte_carlo(xg_grid=xg_grid,xa_grid=xa_grid,saves_grid=saves_grid,other_features=other_features,n_runs=n_runs)

    # Optimization
    priced_pd = add_prices(simulated_pd)
    squad_df = best_player(priced_pd,budget=budget)

    if squad_df is not None:
        print(f"Captain: {squad_df.iloc[squad_df['points'].argmax(),0]}")
        print(squad_df)

    return squad_df

if __name__ == "__main__":
    main(season=26, gameweek=10)

