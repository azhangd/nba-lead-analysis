import numpy as np
import pandas as pd

from lead_model import (
    elo_gaps,
    fit_spread,
    hold_probability,
    lead_needed,
    normal_cdf,
    regulation_steps,
)

TRUE = {
    "A": 2.5,
    "beta": 0.45,
    "kappa": 0.03,
    "per_elo": 0.05,
    "home_elo": 30.0,
    "per_missing": 1.2,
}


def test_fit_recovers_known_spread():
    # Checkpoints whose end-of-regulation margin is drawn from the model with known parameters. A wrong mean or
    # spread formula (e.g. sqrt(t) in place of t**beta, or the shrink on the wrong term) lands far from them.
    rng = np.random.default_rng(0)
    n = 40_000
    s = pd.DataFrame(
        {
            "minutes_left": rng.integers(1, 48, n).astype(float),
            "margin": rng.integers(-25, 26, n),
            "elo_gap": rng.normal(0, 150, n),
            "missing_gap": rng.normal(0, 30, n),
            "is_neutral": rng.random(n) < 0.05,
        }
    )
    t, edge = (
        s.minutes_left,
        s.elo_gap + TRUE["home_elo"] * ~s.is_neutral + TRUE["per_missing"] * s.missing_gap,
    )
    mean = s.margin * (1 - TRUE["kappa"] * np.sqrt(t)) + TRUE["per_elo"] * edge * t / 48
    s["reg_end"] = rng.normal(mean, TRUE["A"] * t ** TRUE["beta"])
    fit = fit_spread(s, {k: TRUE[k] for k in ("per_elo", "home_elo", "per_missing")})
    assert (
        abs(fit["A"] - 2.5) < 0.05
        and abs(fit["beta"] - 0.45) < 0.01
        and abs(fit["kappa"] - 0.03) < 0.003
    )


def test_lead_needed_is_where_hold_probability_reaches_the_level():
    t = np.array([2.0, 6.0, 12.0, 30.0])
    for z, edge in [(2.326, 0.0), (1.645, 120.0), (2.576, -150.0)]:
        lead = lead_needed(z, t, TRUE, edge)
        assert np.allclose(hold_probability(lead, t, TRUE, edge), normal_cdf(z), atol=1e-6)


def test_elo_uses_date_order_not_row_order():
    # The warehouse's Elo sorted MM/DD/YYYY text, so January ran before October. Shuffled rows must not matter.
    games = pd.DataFrame(
        {
            "game_id": ["a", "b", "c"],
            "game_date": pd.to_datetime(["2025-10-22", "2025-11-01", "2026-01-05"]),
            "home": ["OKC", "BOS", "OKC"],
            "away": ["BOS", "OKC", "BOS"],
            "home_score": [120, 99, 110],
            "away_score": [100, 104, 108],
            "is_neutral": False,
        }
    )
    in_order, end = elo_gaps(games)
    shuffled, _ = elo_gaps(games.iloc[[2, 0, 1]])
    assert in_order["a"] == 0 and in_order.equals(shuffled.loc[in_order.index])
    assert (
        abs(sum(end.values()) - 3000) < 1e-9
    )  # every rating point one team gains, the other loses


def test_overtime_line_ends_at_regulation_margin():
    snap = pd.DataFrame({"t": [600.0, 2880.0, 3000.0], "margin": [5, 0, 3]})
    xs, ys = regulation_steps(snap, final_margin=3)
    assert xs[0] == 48 and xs[-1] == 0
    assert ys[-1] == 0  # tied at the end of regulation, not the overtime final
    assert all(a >= b for a, b in zip(xs, xs[1:]))  # minutes left never increases


def test_team_starter_values_finds_only_the_team_with_a_real_gap():
    # Minutes from each team's side where only "BIG" gets 0.15 points a minute per starter and the rest 0.03. A
    # values table with seasons and teams swapped, or no allowance for testing every team, picks the wrong set.
    from lead_model import team_starter_values

    rng = np.random.default_rng(1)
    rows = []
    for season in ("2022-23", "2023-24", "2024-25"):
        for team in ["BIG"] + [f"T{i}" for i in range(9)]:
            n = 40 * 48  # 40 games of 48 minutes
            own, opp = rng.integers(0, 6, n), rng.integers(0, 6, n)
            b = 0.15 if team == "BIG" else 0.03
            rows.append(
                pd.DataFrame(
                    {
                        "season": season,
                        "team": team,
                        "game_id": np.repeat([f"{season}{team}{g}" for g in range(40)], 48),
                        "own": own,
                        "opp": opp,
                        "strength": rng.normal(0, 0.1, n),
                        "margin": rng.normal(0, 10, n),
                        "change": b * own - 0.03 * opp + rng.normal(0, 2.5, n),
                    }
                )
            )
    out = team_starter_values(pd.concat(rows, ignore_index=True), shuffles=50)
    assert set(out["own"]) == {"BIG"} and abs(out["own"]["BIG"] - 0.15) < 0.05
