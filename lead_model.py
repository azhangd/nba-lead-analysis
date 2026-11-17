"""Lead probability model shared by data preparation, the dashboard and the notebooks.

A lead of L points with t minutes left, for a team whose pre-game edge is E Elo points (home court included),
ends regulation as a normal variable with

    mean  L * (1 - kappa * sqrt(t))  +  b * E * t / 48     leads shrink a little; the stronger team gains
    sd    A * t ** beta                                   the margin spreads out a little slower than a random walk

so the lead holds with probability Phi(mean / sd). We use Stern's (1994) random walk with drift,
with two adjustments: the lead shrinkage described by Gabel & Redner (2012), and a spread
that scales as t ** beta rather than sqrt(t).
b and home court are fit to the fit seasons' final margins (as Stern used the point spread), then A, beta and kappa by
maximum likelihood to how far the margin actually moved from each whole-minute checkpoint to the end of regulation.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

REGULATION_S = 2880
# FiveThirtyEight's NBA Elo: K = 20, 100 points of home court, a margin-of-victory multiplier that grows slower
# than the margin and shrinks when the favourite wins. Carry 75% of the deviation from 1500 into the next season.
ELO_K, ELO_HOME, ELO_CARRY = 20.0, 100.0, 0.75


def normal_cdf(x):
    """Standard normal CDF via the Abramowitz-Stegun 7.1.26 erf approximation (max error 1.5e-7).

    The dashboard uses the same approximation in JavaScript.
    """
    z = np.asarray(x, dtype=float) / np.sqrt(2)
    t = 1 / (1 + 0.3275911 * np.abs(z))
    erf_polynomial = (
        (((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592
    ) * t
    return 0.5 * (1 + np.sign(z) * (1 - erf_polynomial * np.exp(-z * z)))


def elo_gaps(games: pd.DataFrame, start: dict | None = None) -> tuple[pd.Series, dict]:
    """Pre-game rating gap (home minus away, home court not included) for every game, in date order.

    games: game_id, game_date, home, away, home_score, away_score, is_neutral (no home court at a neutral site, such
    as the Cup semifinals in Las Vegas or a game abroad). start: last season's final ratings, carried
    over at ELO_CARRY; teams without one start at 1500. Returns the gaps (indexed by game_id) and the final ratings.
    """
    ratings = {
        team: 1500 + ELO_CARRY * (previous_rating - 1500)
        for team, previous_rating in (start or {}).items()
    }
    gaps = {}
    for game in games.sort_values(["game_date", "game_id"]).itertuples():
        home_rating, away_rating = ratings.get(game.home, 1500.0), ratings.get(game.away, 1500.0)
        gaps[game.game_id] = home_rating - away_rating
        adjusted_rating_gap = home_rating + (0.0 if game.is_neutral else ELO_HOME) - away_rating
        final_margin = game.home_score - game.away_score
        expected_home = 1 / (1 + 10 ** (-adjusted_rating_gap / 400))
        winner_edge = adjusted_rating_gap if final_margin > 0 else -adjusted_rating_gap
        margin_multiplier = (abs(final_margin) + 3) ** 0.8 / (7.5 + 0.006 * max(0.0, winner_edge))
        rating_change = ELO_K * margin_multiplier * (float(final_margin > 0) - expected_home)
        ratings[game.home], ratings[game.away] = (
            home_rating + rating_change,
            away_rating - rating_change,
        )
    return pd.Series(gaps, name="elo_gap"), ratings


def minute_states(flow: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Every whole-minute checkpoint inside regulation (not the tip or the buzzer), from the home side.

    flow: game_id, t (seconds elapsed, multiples of 60), margin (home minus away).
    games: game_id, home_score, away_score, elo_gap, missing_gap, is_neutral.
    Adds the end-of-regulation margin and whether the home team won the game.
    """
    game_results = games.set_index("game_id")
    regulation_margins = flow[flow.t == REGULATION_S].set_index("game_id").margin
    checkpoints = flow[(flow.t > 0) & (flow.t < REGULATION_S)].copy()
    checkpoints["minutes_left"] = (REGULATION_S - checkpoints.t) / 60
    checkpoints["reg_end"] = regulation_margins[checkpoints.game_id].to_numpy()
    checkpoints["home_won"] = (game_results.home_score - game_results.away_score)[
        checkpoints.game_id
    ].to_numpy() > 0
    checkpoints["elo_gap"] = game_results.elo_gap[checkpoints.game_id].to_numpy()
    checkpoints["missing_gap"] = game_results.missing_gap[checkpoints.game_id].to_numpy()
    checkpoints["is_neutral"] = game_results.is_neutral[checkpoints.game_id].to_numpy()
    return checkpoints[
        [
            "game_id",
            "minutes_left",
            "margin",
            "reg_end",
            "home_won",
            "elo_gap",
            "missing_gap",
            "is_neutral",
        ]
    ].reset_index(drop=True)


def fit_drift(games: pd.DataFrame) -> dict:
    """Fit the final margin by least squares using the pre-game Elo gap, missing-minutes gap,
    and home court. Neutral sites have no home-court term.

    Return points per Elo point, home court in Elo points, and Elo points per usual minute
    of player absence on the opposing team.
    """
    predictors = np.column_stack(
        [games.elo_gap, games.missing_gap, (~games.is_neutral.astype(bool)).astype(float)]
    )
    (per_elo, points_per_missing_minute, home_court_points), *_ = np.linalg.lstsq(
        predictors, games.home_score - games.away_score, rcond=None
    )
    return {
        "per_elo": float(per_elo),
        "home_elo": float(home_court_points / per_elo),
        "per_missing": float(points_per_missing_minute / per_elo),
    }


def game_edge(d: pd.DataFrame, m: dict):
    """The home side's pre-game edge in Elo points: the rating gap, home court (not at a neutral site), and who's
    missing (usual minutes the away team is without, minus the home team's)."""
    return (
        d.elo_gap + m["home_elo"] * (~d.is_neutral.astype(bool)) + m["per_missing"] * d.missing_gap
    )


def mean_sd(lead, minutes_left, edge, A, beta, kappa, per_elo):
    """Where the margin ends regulation, ~ Normal(mean, sd): a lead `lead` with `minutes_left`, for a team `edge` Elo
    better (home court included)."""
    time_remaining = np.asarray(minutes_left, dtype=float)
    remaining_lead = np.asarray(lead, dtype=float) * (1 - kappa * np.sqrt(time_remaining))
    strength_drift = per_elo * np.asarray(edge, dtype=float) * time_remaining / 48
    mean = remaining_lead + strength_drift
    sd = A * time_remaining**beta
    return mean, sd


def fit_spread(states: pd.DataFrame, drift: dict) -> dict:
    """Maximum-likelihood A, beta, kappa for the move from each checkpoint to the end of regulation, given the drift
    (fit_drift)."""
    edge, per_elo = game_edge(states, drift), drift["per_elo"]

    def negative_log_likelihood(parameters):
        log_scale, beta, kappa = parameters
        scale = np.exp(log_scale)
        mean, sd = mean_sd(states.margin, states.minutes_left, edge, scale, beta, kappa, per_elo)
        standardized_error = (states.reg_end - mean) / sd
        return np.mean(np.log(sd) + 0.5 * standardized_error**2)

    optimization = minimize(
        negative_log_likelihood,
        [np.log(2.5), 0.5, 0.0],
        method="Nelder-Mead",
        options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 10_000},
    )
    if not optimization.success:
        raise RuntimeError(f"spread fit did not converge: {optimization.message}")
    return {
        "A": float(np.exp(optimization.x[0])),
        "beta": float(optimization.x[1]),
        "kappa": float(optimization.x[2]),
    } | drift


def bootstrap_fits(
    games: pd.DataFrame, states: pd.DataFrame, n: int = 100, seed: int = 0
) -> list[dict]:
    """Refit the drift and the spread on n resamples of whole games (with replacement): how much the model's numbers
    move with the games it happened to see. Used for the uncertainty band on the page and the ranges in the appendix.
    """
    rng = np.random.default_rng(seed)
    states_by_game = dict(list(states.groupby("game_id")))
    bootstrap_models = []
    for _ in range(n):
        sampled_game_ids = rng.choice(games.game_id.to_numpy(), len(games))
        drift = fit_drift(games.set_index("game_id").loc[sampled_game_ids].reset_index())
        bootstrap_models.append(
            fit_spread(pd.concat([states_by_game[game_id] for game_id in sampled_game_ids]), drift)
        )
    return bootstrap_models


def starter_minutes(flow: pd.DataFrame, starters: pd.DataFrame) -> pd.DataFrame:
    """Every regulation minute with the change in the home margin over it and each side's starters on the floor at its
    start (starters: game_id, side, t, on, from prep_data.load_starters). Minutes with a count above 5 are dropped:
    those are substitution rows credited to the wrong player."""
    minute_changes = flow[flow.t < REGULATION_S].sort_values(["game_id", "t"]).copy()
    minute_changes["change"] = (
        minute_changes.groupby("game_id").margin.shift(-1) - minute_changes.margin
    )
    minute_changes = minute_changes.dropna(subset=["change"]).astype({"t": float}).sort_values("t")
    starter_events = starters.sort_values("t")
    for side in ("home", "away"):
        side_counts = starter_events[starter_events.side == side][["game_id", "t", "on"]].rename(
            columns={"on": f"{side}_on"}
        )
        minute_changes = pd.merge_asof(
            minute_changes, side_counts, on="t", by="game_id", direction="backward"
        )
    return minute_changes.dropna().query("home_on <= 5 and away_on <= 5").reset_index(drop=True)


def fit_starter_effect(minutes: pd.DataFrame, games: pd.DataFrame, m: dict) -> dict:
    """Estimate the points per minute associated with each additional starter relative to the opponent.

    Regress the minute's margin change on the starter difference, holding constant the total
    starters on the floor, team strength, and current margin (to account for shrinking leads).
    Also return the leading side's average starter difference in the second half when up 10+.
    This is the lineup difference already reflected in the model's curves.
    """
    minute_data = minutes.merge(
        games.assign(edge=game_edge(games, m))[["game_id", "edge"]], on="game_id"
    )
    starter_difference = minute_data.home_on - minute_data.away_on
    starters_on_floor = minute_data.home_on + minute_data.away_on
    strength_per_minute = m["per_elo"] * minute_data.edge / 48
    predictors = np.column_stack(
        [
            starter_difference,
            starters_on_floor,
            strength_per_minute,
            minute_data.margin,
            np.ones(len(minute_data)),
        ]
    )
    per_starter = float(np.linalg.lstsq(predictors, minute_data.change, rcond=None)[0][0])
    rest_decisions = minute_data[
        (minute_data.t >= REGULATION_S / 2) & (minute_data.margin.abs() >= 10)
    ]
    usual_starter_gap = float(
        (np.sign(rest_decisions.margin) * (rest_decisions.home_on - rest_decisions.away_on)).mean()
    )
    return {"per_starter": per_starter, "usual_starter_gap": usual_starter_gap}


def team_minutes(minutes: pd.DataFrame, games: pd.DataFrame, m: dict, season: str) -> pd.DataFrame:
    """starter_minutes from each team's side: the change in its own margin, its starters and the opponent's on the
    floor, its strength term and its margin, so each team's starter value can be fit separately."""
    minute_data = minutes.merge(
        games.assign(edge=game_edge(games, m))[["game_id", "home", "away", "edge"]], on="game_id"
    )
    team_rows = []
    for side, sign in (("home", 1), ("away", -1)):
        own_column, opponent_column = (
            ("home_on", "away_on") if side == "home" else ("away_on", "home_on")
        )
        team_rows.append(
            pd.DataFrame(
                {
                    "season": season,
                    "team": minute_data[side],
                    "game_id": minute_data.game_id,
                    "change": sign * minute_data.change,
                    "own": minute_data[own_column],
                    "opp": minute_data[opponent_column],
                    "strength": sign * m["per_elo"] * minute_data.edge / 48,
                    "margin": sign * minute_data.margin,
                }
            )
        )
    return pd.concat(team_rows, ignore_index=True)


def team_starter_values(d: pd.DataFrame, shuffles: int = 200, seed: int = 0) -> dict:
    """Estimate starter values for each team-season from team_minutes rows.

    Shuffle team labels among team-games within each season to estimate how much the values
    would vary by chance. Compare each team's average across seasons with the league average,
    using a Bonferroni correction for testing every team at 5%. Only teams that pass this check
    receive their own starter value on the dashboard.
    """
    predictors = np.column_stack([d.own, d.opp, d.strength, d.margin, np.ones(len(d))])
    team_game_index = pd.factorize(d.season + " " + d.team + " " + d.game_id)[0]
    # Sum X'X and X'y within each team-game. Shuffles can then reuse these sums.
    team_game_count = team_game_index.max() + 1
    predictor_products = np.zeros((team_game_count, 5, 5))
    predictor_response_products = np.zeros((team_game_count, 5))
    np.add.at(predictor_products, team_game_index, predictors[:, :, None] * predictors[:, None, :])
    np.add.at(
        predictor_response_products, team_game_index, predictors * d.change.to_numpy()[:, None]
    )

    first_row_by_game = pd.Series(np.arange(len(d))).groupby(team_game_index).first().to_numpy()
    team_games = d.iloc[first_row_by_game][["season", "team"]].reset_index(drop=True)

    def slopes(team: pd.Series) -> pd.Series:
        groups = pd.Series(np.arange(len(team_games))).groupby(
            [team_games.season.to_numpy(), team.to_numpy()]
        )
        starter_slopes = {}
        for team_season, game_indices in groups:
            summed_predictors = predictor_products[game_indices].sum(0)
            summed_response = predictor_response_products[game_indices].sum(0)
            coefficients = np.linalg.solve(summed_predictors, summed_response)
            starter_slopes[team_season] = coefficients[0]
        return pd.Series(starter_slopes)

    values = slopes(team_games.team).unstack(0)  # team x season
    rng = np.random.default_rng(seed)
    shuffled_spreads = []
    for _ in range(shuffles):
        shuffled_teams = team_games.groupby("season").team.transform(
            lambda teams: rng.permutation(teams.to_numpy())
        )
        shuffled_spreads.append(slopes(shuffled_teams).std())
    shuffle_spread = np.median(shuffled_spreads)

    team_averages = values.mean(axis=1)
    league_average = values.stack().mean()
    p_values = pd.Series(
        2
        * norm.sf(
            (team_averages - league_average).abs() / (shuffle_spread / np.sqrt(values.shape[1]))
        ),
        index=team_averages.index,
    )
    return {
        "values": values,
        "league": float(league_average),
        "noise": float(shuffle_spread),
        "spread": float(values.stack().std()),
        "own": {
            t: float(team_averages[t])
            for t in team_averages.index
            if p_values[t] * len(team_averages) < 0.05
        },
    }


def hold_probability(lead, minutes_left, m: dict, edge=0.0):
    """Chance a lead of `lead` (negative = trailing) with `minutes_left` holds, for a team `edge` Elo better,
    home court included (vectorized)."""
    mean, sd = mean_sd(lead, minutes_left, edge, m["A"], m["beta"], m["kappa"], m["per_elo"])
    return normal_cdf(mean / sd)


def lead_needed(z: float, minutes_left, m: dict, edge=0.0):
    """The lead at which hold_probability reaches Phi(z): the mean is linear in the lead, so it solves exactly."""
    time_remaining = np.asarray(minutes_left, dtype=float)
    target_margin = z * m["A"] * time_remaining ** m["beta"]
    strength_drift = m["per_elo"] * np.asarray(edge, dtype=float) * time_remaining / 48
    remaining_lead_fraction = 1 - m["kappa"] * np.sqrt(time_remaining)
    return (target_margin - strength_drift) / remaining_lead_fraction


def home_probability(states: pd.DataFrame, m: dict):
    """The model's chance that the home team wins, at every checkpoint."""
    return hold_probability(states.margin, states.minutes_left, m, game_edge(states, m))


def calibration(
    states: pd.DataFrame, m: dict, edges=(0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)
) -> pd.DataFrame:
    """For every checkpoint with a lead: the model's chance the leader wins vs. whether it did. A weaker team's
    small lead can be under 50%, so the bins start at 0."""
    leading_checkpoints = states[states.margin != 0]
    home_win_probability = home_probability(leading_checkpoints, m)
    predicted = np.where(
        leading_checkpoints.margin > 0, home_win_probability, 1 - home_win_probability
    )
    held = np.where(
        leading_checkpoints.margin > 0, leading_checkpoints.home_won, ~leading_checkpoints.home_won
    )
    predictions = pd.DataFrame({"predicted": predicted, "held": held})
    probability_bins = pd.cut(predicted, list(edges))
    summary = predictions.groupby(probability_bins, observed=True).agg(
        leads=("held", "size"),
        predicted=("predicted", "mean"),
        actually_held=("held", "mean"),
    )
    return summary.round(3)


def regulation_steps(snap: pd.DataFrame, final_margin: int) -> tuple[list, list]:
    """Step line of the home margin over regulation, x = minutes left (48 -> 0).

    snap: t (seconds elapsed) and margin after every scoring change. Overtime games end regulation tied, so
    their line ends at the last regulation margin rather than the final score.
    """
    regulation_scores = snap[snap.t <= REGULATION_S]
    elapsed_seconds = [0.0] + regulation_scores.t.tolist()
    margins = [0] + regulation_scores.margin.tolist()
    ending_margin = margins[-1] if (snap.t > REGULATION_S).any() else final_margin
    minutes_left, step_margins = [], []
    for event_index in range(len(elapsed_seconds)):
        if event_index:
            minutes_left.append(48 - elapsed_seconds[event_index] / 60)
            # Hold the previous margin until the score changes.
            step_margins.append(margins[event_index - 1])
        minutes_left.append(48 - elapsed_seconds[event_index] / 60)
        step_margins.append(margins[event_index])
    minutes_left.append(0.0)
    step_margins.append(ending_margin)
    return [round(x, 3) for x in minutes_left], step_margins
