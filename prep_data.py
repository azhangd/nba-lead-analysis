"""Export dashboard tables from nba-analytics-duckdb warehouses, one per season.

Usage: uv run python prep_data.py FIT.duckdb [FIT.duckdb ...] --test TEST.duckdb [TEST.duckdb ...]
  e.g. prep_data.py live-2022-23.duckdb live-2023-24.duckdb live-2024-25.duckdb --test live-2025-26-v2.duckdb live-2026-27.duckdb

The model is fit on the fit seasons together and evaluated on the test seasons. Team ratings
carry over between seasons in date order, including from the last fit season to the first test season.

For every season, export <season>_games.parquet (game results and pre-game strength),
<season>_flow.parquet (the home margin at each whole minute), and <season>_starters.parquet
(starters on the floor). For test seasons, also export <season>_snaps.parquet (the margin
at every scoring change). model.json stores the fitted parameters, 100 bootstrap refits,
and the fit and test seasons. Player names are not exported.
"""

import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

from lead_model import (
    bootstrap_fits,
    elo_gaps,
    fit_drift,
    fit_spread,
    fit_starter_effect,
    minute_states,
    starter_minutes,
    team_minutes,
    team_starter_values,
)

# One checkpoint per minute for fitting; game lines retain every scoring change.
GRID_SECONDS = 60
OUT = Path(__file__).parent / "data"


def season_of(db_path: str) -> str:
    """The one regular season a warehouse holds, from its game ids (002YY...: the season starting in 20YY)."""
    connection = duckdb.connect(db_path, read_only=True)
    years = [
        row[0]
        for row in connection.sql(
            "select distinct substr(game_id, 4, 2) from silver.fct_games "
            "where game_id like '002%'"
        ).fetchall()
    ]
    if len(years) != 1:
        raise ValueError(f"{db_path}: expected one regular season, found {sorted(years)}")
    start_year = int(years[0])
    return f"20{start_year:02d}-{(start_year + 1) % 100:02d}"


# The score after every made shot or free throw, one row per second (the score at the end of that second). Other
# rows' score columns can lag (a late-logged miss or a violation repeats an older score), and free throws in one
# second are sometimes logged out of order, so each team's score is its highest so far.
TIMELINE = """
    with made as (
        select game_id,
               case when period <= 4 then (period - 1) * 720 + 720 - clock_seconds_remaining
                    else 2880 + (period - 5) * 300 + 300 - clock_seconds_remaining end as t,
               score_home::int as h, score_away::int as a
        from silver.fct_play_by_play
        where game_id in (select game_id from games) and action_type in ('2pt', '3pt', 'freethrow')
          and shot_result = 'Made'
    ),
    per_second as (select game_id, t, max(h) as h, max(a) as a from made group by all)
    select game_id, t,
           max(h) over w - max(a) over w as margin
    from per_second
    window w as (partition by game_id order by t rows between unbounded preceding and current row)
    order by game_id, t
"""


def missing_minutes(con, games: pd.DataFrame) -> pd.Series:
    """Return the away team's missing usual minutes minus the home team's, for each game.
    A positive difference favors the home team.

    A player's usual minutes are their average over the team's previous 10 games, with at least
    3 games required. This uses information available before tip-off, when inactive lists and
    starting lineups are public. Without enough early-season history, the contribution is 0.
    """
    box_scores = con.sql("""
        select b.game_id, b.team_id, b.player_id, coalesce(b.minutes, 0) as minutes, b.team_id = g.home_team_id as is_home
        from silver.fct_boxscores b join silver.fct_games g using (game_id)
        where b.game_id in (select game_id from games)
        """).df()
    game_dates = games.set_index("game_id").game_date
    team_games = box_scores[["game_id", "team_id", "is_home"]].drop_duplicates(
        ["game_id", "team_id"]
    )
    team_games = team_games.assign(date=game_dates[team_games.game_id].to_numpy()).sort_values(
        ["team_id", "date", "game_id"]
    )
    team_games["n"] = team_games.groupby("team_id").cumcount()
    # Include each player in every team game, with 0 minutes when they did not play.
    season_rosters = box_scores[["team_id", "player_id"]].drop_duplicates()
    recorded_minutes = box_scores[["game_id", "team_id", "player_id", "minutes"]]
    player_games = team_games.merge(season_rosters, on="team_id").merge(
        recorded_minutes, on=["game_id", "team_id", "player_id"], how="left"
    )
    player_games["minutes"] = player_games.minutes.fillna(0)
    player_games = player_games.sort_values(["team_id", "player_id", "n"])
    player_games["usual"] = player_games.groupby(["team_id", "player_id"]).minutes.transform(
        lambda m: m.shift(1).rolling(10, min_periods=3).mean()
    )
    player_games["missing"] = player_games.usual.fillna(0).where(player_games.minutes == 0, 0)
    missing_by_side = player_games.groupby(["game_id", "is_home"]).missing.sum().unstack()
    return (
        (missing_by_side[False] - missing_by_side[True]).reindex(games.game_id).fillna(0).round(1)
    )


def load_season(db_path: str):
    """Final regular-season games, the home margin each minute, and the exact margin after every scoring change."""
    connection = duckdb.connect(db_path, read_only=True)

    # 1. Regular-season games (ids starting 002) that are final and have been played: a game counts only if it has
    #    play-by-play (some warehouses carry scores for games not played yet).
    games = connection.sql("""
        select game_id, game_date::date as game_date,
               home_team_abbr as home, away_team_abbr as away, home_score, away_score, is_neutral
        from silver.fct_games
        where game_id like '002%' and home_score is not null and away_score is not null
          and game_id in (select game_id from silver.fct_play_by_play)
        order by game_date, game_id
        """).df()
    connection.register("games", games)
    scoring_changes = connection.sql(TIMELINE).df()

    # 2. The margin at any moment is the last scoring change at or before it (an as-of join), up to the end of the
    #    game's last period: regulation is 2,880 s; overtime periods add 300 s each.
    connection.register("snaps", scoring_changes)
    minute_checkpoints = connection.sql(f"""
        with ends as (
            select game_id, 2880 + 300 * greatest(max(period) - 4, 0) as game_end
            from silver.fct_play_by_play where game_id in (select game_id from games) group by 1
        ),
        grid as (select game_id, unnest(range(0, game_end + 1, {GRID_SECONDS})) as t from ends)
        select grid.game_id, grid.t, coalesce(snaps.margin, 0) as margin
        from grid asof left join snaps on grid.game_id = snaps.game_id and grid.t >= snaps.t
        order by grid.game_id, grid.t
        """).df()

    # 3. Pin each game's last grid point to the official final margin, and count the games where the last recorded
    #    score disagreed with it.
    official_margins = games.set_index("game_id").eval("home_score - away_score")
    is_final_checkpoint = (
        minute_checkpoints.groupby("game_id").t.transform("max") == minute_checkpoints.t
    )
    last_checkpoints = minute_checkpoints.loc[is_final_checkpoint].set_index("game_id")
    expected_final_margins = official_margins[
        minute_checkpoints.loc[is_final_checkpoint, "game_id"]
    ].to_numpy()
    score_mismatches = int((last_checkpoints.margin != expected_final_margins).sum())
    minute_checkpoints.loc[is_final_checkpoint, "margin"] = expected_final_margins
    games["missing_gap"] = missing_minutes(connection, games).to_numpy()
    return games, minute_checkpoints, scoring_changes, score_mismatches


def load_starters(db_path: str):
    """How many of each team's five starters were on the floor, from every substitution.

    The play-by-play logs substitutions but not who is on the floor when a period starts, so a starter counts as on
    the floor at the start of a period if their first event in it is anything but checking in (a shot, a foul,
    checking out), and off if it is checking in or they have no event in that period. Returns one row per change
    (game_id, side 'home'/'away', t in seconds elapsed, on = starters on the floor from t), and a check: each
    starter's inferred minutes against the box score's.

    Some games (2025-26's stats_v3 rows) log each substitution as one row with no in/out: the row's player goes out,
    and the one coming in is named only in the text, "SUB: <in> FOR <out>". There the row counts as checking out,
    and the team's player whose name ends with <in> checks in, when exactly one player on that team in that game does
    (the text drops accents, and abbreviates the first name where two players share a surname: "Jal. Williams").
    """
    connection = duckdb.connect(db_path, read_only=True)
    starter_events = connection.sql("""
        with players as (
            select b.game_id, b.player_id, b.team_id, b.player_name, b.minutes, b.started,
                   case when b.team_id = g.home_team_id then 'home' else 'away' end as side
            from silver.fct_boxscores b join silver.fct_games g using (game_id)
            where b.game_id like '002%'
        ),
        starters as (select * from players where started),
        pbp as (
            select *, coalesce(sub_type, '') = '' and action_type = 'substitution' as one_row_sub
            from silver.fct_play_by_play where game_id like '002%'
        ),
        -- one-row substitutions: who came in, by name, kept only where the name points to one player on the team.
        -- The text drops accents (Jokic for Jokić) and adds an initial where two players share a surname (K. Williams)
        one_row as (
            select *, strip_accents(regexp_extract(description, '^SUB: (.+) FOR ', 1)) as nm from pbp where one_row_sub
        ),
        came_in as (
            select p.game_id, p.period, p.action_number, p.clock_seconds_remaining, pl.player_id
            from one_row p join players pl
              on pl.game_id = p.game_id and pl.team_id = p.team_id
             and case when regexp_matches(p.nm, '^[A-Z][a-z]*\\. ')   -- K. Williams, Jal. Williams, St. Curry
                      then starts_with(strip_accents(pl.player_name), regexp_extract(p.nm, '^([A-Z][a-z]*)\\. ', 1))
                           and ends_with(strip_accents(pl.player_name), ' ' || regexp_extract(p.nm, '^[A-Z][a-z]*\\. (.+)$', 1))
                      else ends_with(strip_accents(pl.player_name), p.nm) end
            qualify count(*) over (partition by p.game_id, p.action_number) = 1
        ),
        events as (
            select game_id, player_id, period, action_number, clock_seconds_remaining,
                   case when action_type = 'substitution' then (case when one_row_sub then 'out' else sub_type end)
                   end as sub_kind
            from pbp
            union all
            select game_id, player_id, period, action_number, clock_seconds_remaining, 'in' from came_in
        )
        select s.game_id, s.side, s.player_id, s.minutes, p.period, p.action_number,
               (p.period - 1) * 720 + (720 - p.clock_seconds_remaining) as t_reg,
               case when p.period > 4 then 2880 + (p.period - 5) * 300 + (300 - p.clock_seconds_remaining) end as t_ot,
               p.sub_kind
        from starters s join events p on p.game_id = s.game_id and p.player_id = s.player_id
        order by s.game_id, s.player_id, p.period, p.action_number, p.sub_kind = 'in'
        """).df()
    starter_events["t"] = starter_events.t_ot.fillna(starter_events.t_reg)
    periods = connection.sql(
        "select game_id, max(period) as last from silver.fct_play_by_play group by 1"
    ).df()
    last_period = dict(zip(periods.game_id, periods["last"]))

    def period_start(period):
        return (period - 1) * 720 if period <= 4 else 2880 + (period - 5) * 300

    def period_end(period):
        return period_start(period) + (720 if period <= 4 else 300)

    changes, minute_checks = [], []
    for (game_id, side, player_id), player_events in starter_events.groupby(
        ["game_id", "side", "player_id"], sort=False
    ):
        seconds_played = 0.0
        for period in range(1, last_period[game_id] + 1):
            period_events = player_events[player_events.period == period]
            on_floor = len(period_events) > 0 and period_events.sub_kind.iloc[0] != "in"
            entered_at = period_start(period)
            if on_floor:
                changes.append((game_id, side, entered_at, +1))
            for event_time, substitution in zip(period_events.t, period_events.sub_kind):
                if substitution == "in" and not on_floor:
                    on_floor, entered_at = True, event_time
                    changes.append((game_id, side, event_time, +1))
                elif substitution == "out" and on_floor:
                    on_floor = False
                    seconds_played += event_time - entered_at
                    changes.append((game_id, side, event_time, -1))
            if on_floor:
                seconds_played += period_end(period) - entered_at
                changes.append((game_id, side, period_end(period), -1))
        minute_checks.append(
            (game_id, player_id, seconds_played / 60, player_events.minutes.iloc[0])
        )

    count_changes = (
        pd.DataFrame(changes, columns=["game_id", "side", "t", "delta"])
        .groupby(["game_id", "side", "t"])
        .delta.sum()
    )
    timeline = count_changes.groupby(level=[0, 1]).cumsum().rename("on").reset_index()
    return timeline, pd.DataFrame(
        minute_checks, columns=["game_id", "player_id", "inferred_min", "box_min"]
    )


def main(fit_dbs: list[str], test_dbs: list[str]) -> None:
    OUT.mkdir(exist_ok=True)
    warehouses_by_season = {
        season_of(database_path): database_path for database_path in (*fit_dbs, *test_dbs)
    }
    if len(warehouses_by_season) != len(fit_dbs) + len(test_dbs):
        raise ValueError(
            f"each warehouse must hold a different season, got {list(warehouses_by_season)}"
        )
    fit_seasons, test_seasons = sorted(
        season_of(database_path) for database_path in fit_dbs
    ), sorted(season_of(database_path) for database_path in test_dbs)
    if fit_seasons[-1] >= test_seasons[0]:
        raise ValueError(
            f"the test seasons must come after the fit seasons, got fit {fit_seasons}, test {test_seasons}"
        )
    season_data = {
        season: load_season(warehouses_by_season[season]) for season in sorted(warehouses_by_season)
    }  # in date order, for the Elo below

    # Rebuild Elo in date order to avoid the warehouse’s text-date sorting error, and carry ratings between seasons.
    ratings = None
    for season, (
        games,
        minute_checkpoints,
        scoring_changes,
        score_mismatches,
    ) in season_data.items():
        gaps, ratings = elo_gaps(games, start=ratings)
        games["elo_gap"] = gaps[games.game_id].to_numpy().round(1)
        games.to_parquet(OUT / f"{season}_games.parquet", index=False)
        minute_checkpoints.to_parquet(OUT / f"{season}_flow.parquet", index=False)
        overtime_games = int((minute_checkpoints.groupby("game_id").t.max() > 2880).sum())
        print(
            f"{season}  scores:   {len(games):>5,} games, {games.game_date.min():%Y-%m-%d} to "
            f"{games.game_date.max():%Y-%m-%d}, {overtime_games} to overtime, {len(minute_checkpoints):,} minute checkpoints, "
            f"{score_mismatches} final score(s) pinned to the official one"
        )

    # Reconstruct starter counts for all seasons and check them against box-score minutes.
    # Test seasons also need every scoring change for the game charts.
    for season in (*fit_seasons, *test_seasons):
        if season in test_seasons:
            season_data[season][2].to_parquet(OUT / f"{season}_snaps.parquet", index=False)
        starters, starter_check = load_starters(warehouses_by_season[season])
        starters.to_parquet(OUT / f"{season}_starters.parquet", index=False)
        minute_difference = (starter_check.inferred_min - starter_check.box_min).abs()
        print(
            f"{season}  starters: {len(starter_check):>5,} starter games, minutes within 30 s of the box score in "
            f"{(minute_difference <= 0.5).mean():.1%}, within 2 min in {(minute_difference <= 2).mean():.1%}"
        )

    # 6. Fit on the fit seasons only: drift from final margins, then the spread and shrink from every checkpoint.
    fit_games = pd.concat([season_data[season][0] for season in fit_seasons], ignore_index=True)
    fit_flow = pd.concat([season_data[season][1] for season in fit_seasons], ignore_index=True)
    drift = fit_drift(fit_games)
    states = minute_states(fit_flow, fit_games)
    fit_label = (
        fit_seasons[0] if len(fit_seasons) == 1 else f"{fit_seasons[0]} to {fit_seasons[-1]}"
    )  # a label for the page
    model = fit_spread(states, drift) | {
        "fit_season": fit_label,
        "fit_seasons": fit_seasons,
        "fit_games": len(fit_games),
        "fit_checkpoints": len(states),
        "test_seasons": test_seasons,
    }
    # Fit the starter-count adjustment used in the dashboard’s lineup comparison.
    starter_minutes_by_season = {
        season: starter_minutes(
            season_data[season][1], pd.read_parquet(OUT / f"{season}_starters.parquet")
        )
        for season in fit_seasons
    }
    model |= fit_starter_effect(pd.concat(starter_minutes_by_season.values()), fit_games, model)
    # Use team-specific starter estimates only where the comparison supports them.
    team_minutes_by_season = []
    for season in fit_seasons:
        season_team_minutes = team_minutes(
            starter_minutes_by_season[season], season_data[season][0], model, season
        )
        team_minutes_by_season.append(season_team_minutes)
    team_effects = team_starter_values(pd.concat(team_minutes_by_season))
    model["team_per_starter"] = team_effects["own"]
    # Estimate model uncertainty with 100 resamples of whole games.
    model["boot"] = bootstrap_fits(fit_games, states)
    (OUT / "model.json").write_text(json.dumps(model, indent=1))
    print(
        f"model fit on {fit_label}: A = {model['A']:.3f}, beta = {model['beta']:.3f}, kappa = {model['kappa']:.4f}, "
        f"{100 * model['per_elo']:.2f} pts per 100 Elo, home court {model['home_elo']:.0f} Elo"
    )
    print(
        f"  {model['per_missing'] * model['per_elo']:.3f} pts per usual minute missing, a starter on the floor "
        f"{model['per_starter']:.3f} pts a minute, own values: "
        f"{ {t: round(v, 3) for t, v in model['team_per_starter'].items()} }"
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--test" not in args or args.index("--test") == 0 or args[-1] == "--test":
        sys.exit(__doc__)
    cut = args.index("--test")
    main(args[:cut], args[cut + 1 :])
