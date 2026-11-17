"""Build the standalone NBA lead dashboard from data/ into docs/index.html.

One HTML file with BokehJS inlined; every interaction is a CustomJS callback on data already in the
page, so it needs no Python server and no network.

The module defines the display settings and data summaries first. build() then loads
season data, creates the controls and charts, connects the browser callbacks, and
assembles the page. CustomJS argument names are the bridge between Python objects
and JavaScript variables; the data-source column names are shared by both.
"""

import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from bokeh.document import Document
from bokeh.embed import file_html
from bokeh.events import MouseLeave, Tap
from bokeh.layouts import column, row
from bokeh.models import (
    BasicTicker,
    BoxAnnotation,
    Title,
    Button,
    ColumnDataSource,
    CustomJS,
    Div,
    FactorRange,
    FixedTicker,
    HoverTool,
    Label,
    LabelSet,
    Legend,
    LegendItem,
    CheckboxButtonGroup,
    InlineStyleSheet,
    RadioButtonGroup,
    Range1d,
    Select,
    Span,
    Spinner,
    TapTool,
    BoxZoomTool,
    PanTool,
    ResetTool,
    WheelZoomTool,
    TextInput,
)
from bokeh.plotting import figure
from bokeh.resources import INLINE
from bokeh.themes import Theme

from lead_model import game_edge, hold_probability, lead_needed, regulation_steps

HERE = Path(__file__).parent
DATA = HERE / "data"
MODEL = json.loads((DATA / "model.json").read_text())
FIT_SEASONS, TEST_SEASONS = (
    MODEL["fit_season"],
    MODEL["test_seasons"],
)  # the fit-season range and the held-out test seasons
ALL_TEAMS = "All teams"
# Start with one selected game.
START_SEASON, START_TEAM, START_GAME = "2026-27", "LAL", "TOR @ LAL, Nov 01"
# sizes: the page is ~1,240 px wide so it fits a 1,280-px laptop screen; charts in a row share a height
CURVES_HEIGHT = 560  # as tall as the control panel beside it
PANEL_WIDTH = 430  # the control panel's content: the five matchup buttons fit without spilling
# Team bars sit beside the game chart, starter strip and lower chart row.
BARS_WIDTH, GAP = 280, 16
LINES_WIDTH, LINES_HEIGHT = 1240 - BARS_WIDTH - GAP, 420
SLOPE_WIDTH, LOWER_ROW_HEIGHT = 300, 300
STRIP_HEIGHT = 190  # the picked game's starters on the floor, under the game chart
BOTTOM_HEIGHT = (
    LINES_HEIGHT + STRIP_HEIGHT + LOWER_ROW_HEIGHT + 20
)  # the team bars: as tall as the three rows (20: the column's gaps)

# Outcomes after the probability threshold: held leads are muted so color draws the eye
# only to what went wrong: amber = a close call but won, red = lost. Checked: the four outcome colors (with "never safe
# enough" gray) stay at least 24 ΔE apart under deuteranopia and protanopia; the darker text versions are 5:1+ on white.
HELD, SCARE, BLOWN = (
    "#7c97b8",
    "#cb840e",
    "#cf3a1f",
)  # each at least 3:1 on white (WCAG 1.4.11, graphics)
HELD_TEXT, SCARE_TEXT, BLOWN_TEXT = "#4d6789", "#9a6400", "#b0301a"
INK, INK2, MUTED, FADED, GRID, BG = (
    "#111111",
    "#474747",
    "#6e6e6e",
    "#c7c7c7",
    "#e8e8e8",
    "#ffffff",
)
LEADS = [3, 6, 10, 15, 20]  # reference curves, light to dark
LEAD_COLORS = [
    "#cfcfcf",
    "#b0b0b0",
    "#8f8f8f",
    "#6e6e6e",
    "#4d4d4d",
]  # grays: color is kept for outcomes
REST_THRESHOLDS = [96.0, 97.0, 98.0, 99.0, 99.5]  # rest-threshold choices (percent safe)
CLOSE_CALL_THRESHOLD = 95.0  # A qualifying lead becomes a close call below this probability.
DECISION_MIN = 2  # Exclude the final two minutes from threshold analysis: possession and fouling are not modeled.
MATCHUP_KEYS = [
    "much-weaker",
    "weaker",
    "even",
    "stronger",
    "much-stronger",
]  # their names in the page's address
MATCHUPS = [
    ("Much weaker", -150),
    ("Weaker", -75),
    ("Even", 0),
    ("Stronger", 75),
    ("Much stronger", 150),
]  # Elo
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
# one-sentence definitions, shown on hover wherever the term appears (for readers new to basketball)
GLOSSARY = {
    "starters": "The five players who start the game, usually the team's best.",
    "lead": "How many points the team is ahead by.",
    "close call": "A lead reached the selected probability threshold, then fell below 95% before the final two minutes. This describes the model estimate, not a substitution recommendation.",
    "matchup": "How much better one team is than the other, from its results so far (Elo ratings).",
    "safe": "The chance the team that's ahead still wins.",
}
SHOW_LABELS = [
    "Held",
    "Close call",
    "Lost",
    "Below threshold",
]  # the game filter, in outcome order
MIN_TICKS = {
    48: "Start",
    36: "2nd qtr",
    24: "Halftime",
    12: "4th qtr",
    6: "6",
    2: "2",
    0: "End",
}  # 4 quarters of 12 min

THEME = Theme(
    json={
        "attrs": {
            "Plot": {
                "background_fill_color": BG,
                "border_fill_color": BG,
                "outline_line_color": None,
            },
            "Title": {
                "text_font": FONT,
                "text_font_size": "14px",
                "text_color": INK,
                "text_font_style": "bold",
            },
            "Axis": {
                "axis_line_color": FADED,
                "major_tick_line_color": FADED,
                "minor_tick_line_color": None,
                "major_label_text_color": INK2,
                "major_label_text_font": FONT,
                "major_label_text_font_size": "12px",
                "axis_label_text_color": INK2,
                "axis_label_text_font": FONT,
                "axis_label_text_font_style": "normal",
                "axis_label_text_font_size": "12px",
            },
            "Grid": {"grid_line_color": GRID, "grid_line_alpha": 0.6},
            "Legend": {
                "label_text_font": FONT,
                "label_text_font_size": "11px",
                "label_text_color": INK2,
                "background_fill_alpha": 0.85,
                "border_line_color": None,
                "glyph_width": 16,
                "glyph_height": 12,
                "label_height": 12,
                "spacing": 2,
                "padding": 6,
                "margin": 6,
            },
        }
    }
)
PANEL_BG, PANEL_EDGE = (
    "#ffffff",
    "#e4e4e7",
)  # cards: white with a thin neutral edge, no tint
OFF = (
    [-99, -98],
    [0, 0],
)  # legend glyphs' data: past the charts' fixed x ranges (24 or 48 to 0), so never drawn
CLICKABLE = InlineStyleSheet(
    css="* { cursor: pointer !important; }"
)  # charts you pick from: lines, slopes, bars
# The curves' y-axis is spaced by the chance of LOSING the lead (log scale), so each step toward "safe" gets the same
# room: 90 -> 99 -> 99.9% are equal steps, and the probability thresholds (96-99.5%) are no longer squeezed into the top 5%.
SAFE_TICKS = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
LABEL_X = 22.2  # minutes left where the "up N" labels sit: between the circle columns at 24 and 20
Y_CAP = 3.4  # 99.96% safe: beyond the chart's top, so near-certain leads simply leave the frame


def outcome_filter_css() -> InlineStyleSheet:
    """Style the outcome filters to match the chart colors."""
    css = f""".bk-btn-group {{ gap: 8px; flex-wrap: wrap; }}
.bk-btn-group > .bk-btn, .bk-btn-group > .bk-btn.bk-active {{ background: white; border: 1px solid {PANEL_EDGE}; box-shadow: none; border-radius: 4px !important;
  padding: 6px; color: {INK}; font-weight: 400; }}
.bk-btn-group > .bk-btn:not(.bk-active) {{ color: {MUTED}; }}
.bk-btn-group > .bk-btn {{ flex: none; cursor: pointer; }}
.bk-btn-group > .bk-btn:hover {{ color: {INK}; border-color: #8794a3; }}
.bk-btn-group > .bk-btn:focus:not(:focus-visible) {{ outline: none; }}
.bk-btn::before {{ content: ''; display: inline-block; width: 13px; height: 13px; border-radius: 2px; margin-right: 6px;
  vertical-align: -2px; border: 1.5px solid; text-align: center; line-height: 13px; font-size: 11px; }}
.bk-btn.bk-active::before {{ content: "✓"; color: {INK}; }}"""
    for outcome_index, color in enumerate((HELD, SCARE, BLOWN, FADED), start=1):
        css += f"\n.bk-btn:nth-child({outcome_index})::before {{ border-color: {color}; }} .bk-btn.bk-active:nth-child({outcome_index})::before {{ background: {color}; }}"
    return InlineStyleSheet(css=css)


def zoomable(plot):
    """Drag a box to zoom; pan and scroll-wheel zoom are in the toolbar, off by default so the page still scrolls."""
    box_zoom = BoxZoomTool()
    plot.add_tools(box_zoom, PanTool(), WheelZoomTool(), ResetTool())
    plot.toolbar.active_drag, plot.toolbar.active_scroll = box_zoom, None
    plot.toolbar.logo = None
    plot.toolbar.autohide = True


def how_to_read(plot, text):
    """A plain-language line under a chart's title, for readers who don't follow basketball (or this chart type)."""
    plot.add_layout(
        Title(
            text=text,
            text_font=FONT,
            text_font_size="12px",
            text_font_style="normal",
            text_color=MUTED,
        ),
        "above",
    )


def spaced(probability):
    """Where a hold probability sits on the curves' y-axis: -log10(chance of losing), capped."""
    return np.minimum(
        -np.log10(np.clip(1 - np.asarray(probability, dtype=float), 1e-12, None)), Y_CAP
    )


# One stylesheet for every text block and widget (each lives in its own shadow DOM, so each gets it). Type scale:
# 22 title, 17 section heads, 14 body and chart titles, 13 controls, 12 notes and axes, 11 dense chart labels;
# numbers lead: 28 for the three findings, 40 for the answer's percentage. Widgets otherwise fall back to Helvetica/Arial.
PAGE = InlineStyleSheet(css=f"""
:host {{ font-family: {FONT}; font-size: 14px; line-height: 1.45; color: {INK2}; font-variant-numeric: tabular-nums; }}
label, .bk-input, .bk-btn {{ font-family: {FONT}; font-size: 13px; }}
label, .label {{ font-size: 13px; font-weight: 600; color: {INK}; }}
.bk-input, .bk-btn {{ box-sizing: border-box; min-height: 32px; border-radius: 4px; }}
.bk-input {{ height: 32px !important; }}
.bk-input:hover, .bk-btn:hover {{ border-color: #8794a3; }}
.bk-input:focus-visible, .bk-btn:focus-visible {{ outline: 2px solid #3e5d7c; outline-offset: 2px; }}
.bk-btn-default.bk-active {{ background: {INK}; color: white; border-color: {INK}; }}
h1 {{ font-size: 24px; color: {INK}; margin: 4px 0 10px; }}
h2 {{ font-size: 18px; color: {INK}; margin: 18px 0 4px; padding-top: 18px; border-top: 1px solid {PANEL_EDGE}; width: min(1230px, calc(100vw - 42px)); }}
b, .ink {{ color: {INK}; }}
.big {{ font-size: 36px; letter-spacing: -0.5px; font-weight: 700; line-height: 1.1; color: {INK}; margin: 4px 0; }}
.small {{ font-size: 12px; }}
.space {{ margin-top: 8px; }}
.muted {{ color: {MUTED}; }}
@media (max-width: 650px) {{ .tiles {{ flex-direction: column; gap: 16px; }} .tile {{ border-left: none; padding: 0; }} }}
.tiles {{ display: flex; width: min(1230px, calc(100vw - 42px)); margin-top: 6px; padding: 12px 16px; box-sizing: border-box; background: transparent; border: 1px solid #bbbbbd; border-radius: 8px; }}
.tile {{ flex: 1; padding: 2px 16px; border-left: 1px solid {PANEL_EDGE}; }}
.tile:first-child {{ border-left: none; padding-left: 0; }}
.num {{ display: block; font-size: 24px; letter-spacing: -0.3px; font-weight: 700; line-height: 1.2; color: {INK}; }}
.held {{ color: {HELD_TEXT}; }}
abbr[title] {{ text-decoration: underline dotted {MUTED}; text-underline-offset: 3px; cursor: help; }}
.verdict {{ font-size: 15px; font-weight: 600; margin: 10px 0 0; }}
.verdict.muted {{ color: {INK2}; }}
.answer {{ display: grid; gap: 16px; }}
.answer .verdict {{ margin: 0 0 4px; }}
.answer .big {{ margin: 0; }}
.answer details {{ margin: 0; }}
.comparison {{ display: flex; align-items: baseline; gap: 12px; padding-top: 12px; border-top: 1px solid {PANEL_EDGE}; }}
.comparison b {{ font-size: 14px; white-space: nowrap; }}
details {{ margin-top: 6px; }}
summary {{ cursor: pointer; color: {MUTED}; }}
.intro {{ max-width: 860px; font-size: 14px; color: {INK2}; margin: 2px 0 6px; }}
.game {{ width: 100%; box-sizing: border-box; margin: 0; }}
.game details {{ margin: 0; }}
.game summary {{ font-size: 13px; }}
.cap {{ font-size: 14px; color: {INK2}; }}
.story {{ font-size: 13px; color: {INK2}; margin: 6px 0 0; max-width: 1000px; }}
.scare {{ color: {SCARE_TEXT}; }}
.lost {{ color: {BLOWN_TEXT}; }}
/* the team and game lines: short labelled fields in a row, read like the slope chart's three moments */
.facts {{ display: flex; flex-wrap: wrap; align-items: flex-end; column-gap: 16px; row-gap: 4px; margin: 6px 0 2px; }}
.game .facts {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); column-gap: 16px; }}
.game .fact .v {{ white-space: normal; font-size: 13px; }}
.game .fact b {{ font-weight: 500; }}
@media (max-width: 1000px) {{ .game .facts {{ grid-template-columns: repeat(2, minmax(0, 1fr)); row-gap: 12px; }} .game .fact.head {{ grid-column: 1 / -1; }} }}
.fact {{ display: flex; flex-direction: column; }}
.fact .k {{ font-size: 11px; color: {MUTED}; line-height: 1.3; }}
.fact .v {{ font-size: 14px; color: {INK}; white-space: nowrap; }}
.fact.head .v {{ font-weight: 700; }}
.fact.hint .v {{ font-size: 12px; color: {MUTED}; }}
""")


def checkpoints(model: dict) -> pd.DataFrame:
    """Collect minute checkpoints with a lead from the test seasons, which were excluded from fitting.

    Each row contains the lead, minutes remaining, whether the leading team won, and its
    Elo edge including home court. These rows let us compare model estimates with outcomes.
    """
    season_checkpoints = []
    for season in TEST_SEASONS:
        season_games = pd.read_parquet(DATA / f"{season}_games.parquet")
        season_flow = pd.read_parquet(DATA / f"{season}_flow.parquet")
        game_results = season_games.assign(
            final=season_games.home_score - season_games.away_score,
            home_edge=game_edge(season_games, model),
        )[["game_id", "final", "home_edge"]]
        season_checkpoints.append(season_flow.merge(game_results, on="game_id"))
    all_checkpoints = pd.concat(season_checkpoints)
    all_checkpoints = all_checkpoints[
        (all_checkpoints.margin != 0) & (all_checkpoints.t < 2880)
    ]  # not the final buzzer, where every lead has trivially held
    leading_side = np.sign(all_checkpoints.margin)
    return all_checkpoints.assign(
        left=(2880 - all_checkpoints.t) / 60,
        lead=all_checkpoints.margin.abs(),
        held=leading_side == np.sign(all_checkpoints.final),
        edge=leading_side * all_checkpoints.home_edge,
    )


def near_checkpoints(checkpoint_data: pd.DataFrame, lead: int, minutes_left: float) -> pd.DataFrame:
    """Select games with leads from lead-1 to lead+1 and times from minutes_left-1 to minutes_left+1.

    Count each game once, using its checkpoint nearest minutes_left. Choosing the earliest
    checkpoint instead would put the comparison about 0.3 minutes early on average.
    """
    nearby = checkpoint_data[
        checkpoint_data.lead.between(lead - 1, lead + 1)
        & checkpoint_data.left.between(minutes_left - 1, minutes_left + 1)
    ]
    return (
        nearby.assign(off=(nearby.left - minutes_left).abs())
        .sort_values(["off", "t"])
        .drop_duplicates("game_id")
    )


def observed_holds(model: dict) -> tuple[dict, int]:
    """For every lead (1-40) and minutes left (0.5-24, by half minutes): [games, games the leading team won], from
    near_checkpoints. Real games mix every matchup; against each game's own matchup the model moves by under 4 points
    (leading teams average 37-116 Elo better, more for bigger leads), so the circles sit fairly beside the even curves.
    """
    test_checkpoints = checkpoints(model)
    observed_counts = {}
    for lead in range(1, 41):
        for minutes_left in np.arange(0.5, 24.01, 0.5):
            nearby = near_checkpoints(test_checkpoints, lead, minutes_left)
            if len(nearby):
                observed_counts[f"{lead}|{minutes_left:g}"] = [len(nearby), int(nearby.held.sum())]
    return observed_counts, test_checkpoints.game_id.nunique()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a share k/n that stays inside 0-1 (unlike the plain one) when the share is near 100%."""
    observed_rate = k / n
    z_squared = z * z
    denominator = 1 + z_squared / n
    center = (observed_rate + z_squared / (2 * n)) / denominator
    half_width = (
        z * np.sqrt(observed_rate * (1 - observed_rate) / n + z_squared / (4 * n * n)) / denominator
    )
    return center - half_width, center + half_width


def season_rule_costs(model: dict, season: str) -> dict:
    """Estimate the season-level outcomes for each probability threshold from whole-minute checkpoints.

    Count the games in which each team could rest its starters, the minutes remaining then,
    and how many of those games were lost. Close calls are excluded because minute sampling
    misses about half of them, as checked against this season's basket-by-basket data.
    """
    games = pd.read_parquet(DATA / f"{season}_games.parquet")
    flow = pd.read_parquet(DATA / f"{season}_flow.parquet")
    flow = flow[flow.t < 2880].assign(left=lambda d: (2880 - d.t) / 60)
    checkpoints_by_game = {
        g: (d.left.to_numpy(), d.margin.to_numpy()) for g, d in flow.groupby("game_id")
    }
    team_count = len(set(games.home) | set(games.away))
    threshold_costs = {}
    for rest_threshold in REST_THRESHOLDS:
        z, rest_minutes, lost_leads = NormalDist().inv_cdf(rest_threshold / 100), [], 0
        for game_id, home_edge, final in zip(
            games.game_id, game_edge(games, model), games.home_score - games.away_score
        ):
            left, margin = checkpoints_by_game[game_id]
            for side in (1, -1):
                lead = side * margin
                required_lead = lead_needed(z, np.maximum(left, 1e-6), model, side * home_edge)
                safe_indices = np.flatnonzero(
                    (left >= DECISION_MIN) & (lead > 0) & (lead >= required_lead)
                )
                if len(safe_indices):
                    rest_minutes.append(left[safe_indices[0]])
                    lost_leads += side * final < 0
        threshold_costs[f"{rest_threshold:.1f}"] = dict(
            games=len(games),
            per_team=len(rest_minutes) / team_count,
            minutes=float(np.mean(rest_minutes)),
            lost=int(lost_leads),
            lost_per_team=lost_leads / team_count,
        )
    return threshold_costs


def starters_on_floor(season: str) -> dict:
    """For each game of a test season, the number of each side's starters on the floor through regulation, as step
    points in minutes left: {"home": [xs, ys], "away": [xs, ys]} (prep_data.load_starters)."""
    starter_events = pd.read_parquet(DATA / f"{season}_starters.parquet")
    starter_events = starter_events[starter_events.t < 2880].sort_values("t")
    events_by_game_side = {
        k: team_events for k, team_events in starter_events.groupby(["game_id", "side"])
    }
    starter_counts = {}
    for game_id in starter_events.game_id.unique():
        starter_counts[game_id] = {}
        for side in ("home", "away"):
            team_events = events_by_game_side.get((game_id, side), starter_events.iloc[:0])
            # Match regulation_steps rounding so events at the same moment line up.
            minutes_left = [round((2880 - t) / 60, 3) for t in team_events.t] + [0.0]
            starter_counts[game_id][side] = [
                minutes_left,
                [int(v) for v in team_events.on]
                + [int(team_events.on.iloc[-1]) if len(team_events) else 0],
            ]
    return starter_counts


def build() -> str:
    season_games = []
    scoring_changes_by_game = {}
    on_floor = {}
    for season in TEST_SEASONS:
        season_games.append(pd.read_parquet(DATA / f"{season}_games.parquet").assign(season=season))
        scoring_changes = pd.read_parquet(DATA / f"{season}_snaps.parquet")
        for game_id, game_scoring_changes in scoring_changes.groupby("game_id"):
            scoring_changes_by_game[game_id] = game_scoring_changes
        on_floor.update(starters_on_floor(season))
    games = pd.concat(season_games, ignore_index=True)
    model = MODEL
    model_parameters = {
        k: model[k]
        for k in (
            "A",
            "beta",
            "kappa",
            "per_elo",
            "per_starter",
            "usual_starter_gap",
            "team_per_starter",
        )
    }
    observed_counts, observed_game_count = observed_holds(model)
    cost_season = TEST_SEASONS[0]  # the earliest test season: a whole one
    final = (games.home_score - games.away_score).tolist()

    game_steps = [
        regulation_steps(scoring_changes_by_game[g], f) for g, f in zip(games.game_id, final)
    ]
    game_labels = [
        f"{a} @ {h}, {d:%b %d}: {a} {as_}–{hs} {h}{' (OT)' if (scoring_changes_by_game[gid].t > 2880).any() else ''}"
        for gid, a, h, d, as_, hs in zip(
            games.game_id,
            games.away,
            games.home,
            games.game_date,
            games.away_score,
            games.home_score,
        )
    ]
    teams = sorted(set(games.home) | set(games.away))
    overtime_games = [bool((scoring_changes_by_game[gid].t > 2880).any()) for gid in games.game_id]
    game_data = dict(
        date=[f"{d:%b %d}" for d in games.game_date],
        hs=games.home_score.tolist(),
        as_=games.away_score.tolist(),
        ot=overtime_games,
        id=games.game_id.tolist(),
        season=games.season.tolist(),
        home=games.home.tolist(),
        away=games.away.tolist(),
        final=final,
        label=game_labels,
        edge=game_edge(games, model)
        .round(1)
        .tolist(),  # home side's edge: rating, home court, who's missing
        xs=[s[0] for s in game_steps],
        ys=[s[1] for s in game_steps],
    )
    game_data["st"] = [on_floor[g] for g in games.game_id]
    threshold_z_scores = {
        f"{p:.1f}": NormalDist().inv_cdf(p / 100) for p in REST_THRESHOLDS + [CLOSE_CALL_THRESHOLD]
    }
    initial_game_index = next(
        i
        for i, (lab, s) in enumerate(zip(game_labels, games.season))
        if s == START_SEASON and lab.startswith(START_GAME)
    )
    # minutes left, read left to right: the curves cover the second half, where rest decisions happen

    # ---------- controls ----------
    # no low/high: Bokeh keeps showing an out-of-range entry without changing the value, so the box and the answer
    # would disagree. The callback cleans every entry (range, rounding, empty) and writes it back instead; tags hold
    # the last good value for an emptied box.
    widget_style = dict(stylesheets=[PAGE])
    lead_input = Spinner(
        title="Points ahead", step=1, value=12, width=140, tags=[12], **widget_style
    )
    # time left as a game clock: "4:30" or "4.5" both work; the callback rounds to half minutes and writes "4:30" back
    time_input = TextInput(title="Time left", value="4:00", width=140, tags=[4], **widget_style)
    matchup_title = Div(
        text='<div class="label">Matchup</div>',
        width=PANEL_WIDTH,
        **widget_style,
    )
    matchup_control = RadioButtonGroup(
        labels=[name for name, _ in MATCHUPS],
        active=2,
        width=PANEL_WIDTH,
        **widget_style,
    )
    matchup_note = Div(
        width=PANEL_WIDTH, **widget_style
    )  # what the matchup means in points
    level_title = Div(text='<div class="label">Probability threshold</div>', **widget_style)
    rest_threshold = RadioButtonGroup(
        labels=[f"{r:g}%" for r in REST_THRESHOLDS],
        active=REST_THRESHOLDS.index(99.0),
        width=PANEL_WIDTH,
        **widget_style,
    )
    wrapping_choices = InlineStyleSheet(css="""
@media (max-width: 500px) {
  .bk-btn-group { flex-wrap: wrap; }
  .bk-btn-group > .bk-btn { flex: 1 0 auto; }
}
""")
    for choices in (matchup_control, rest_threshold):
        choices.stylesheets = [PAGE, wrapping_choices]
    level_note = Div(width=PANEL_WIDTH, **widget_style)
    scenario_context = Div(width=PANEL_WIDTH, **widget_style)
    answer = Div(width=PANEL_WIDTH, **widget_style)
    input_notice = Div(width=PANEL_WIDTH, visible=False, **widget_style)
    season_select = Select(
        title="Season",
        value=START_SEASON,
        width=150,
        **widget_style,
        options=[
            (s, f"{s}{'' if s == TEST_SEASONS[0] else ' so far'}") for s in reversed(TEST_SEASONS)
        ],
    )
    season_select.visible = len(TEST_SEASONS) > 1
    team_select = Select(
        title="Team",
        value=START_TEAM,
        options=[ALL_TEAMS] + teams,
        width=160,
        **widget_style,
    )
    game_pick = Select(
        title="Game",
        value="-1",
        options=[("-1", "Pick a game")],
        width=440,
        **widget_style,
    )
    # step through the menu's games in its order (tags: the game indexes, set in update)
    clear_game = Button(label="Clear selection", width=112, disabled=True, **widget_style)
    clear_game.js_on_event(
        "button_click", CustomJS(args=dict(pick=game_pick), code='pick.value = "-1";')
    )
    prev_game = Button(label="◀ Previous", width=96, **widget_style)
    next_game = Button(label="Next ▶", width=80, **widget_style)
    navigation_style = InlineStyleSheet(
        css=".bk-input, .bk-btn { height: 32px !important; box-sizing: border-box; }"
    )
    for widget in (game_pick, clear_game, prev_game, next_game):
        widget.stylesheets = [PAGE, navigation_style]
    reset = Button(
        label="Reset all",
        width=90,
        stylesheets=[
            PAGE,
            InlineStyleSheet(
                css=f".bk-btn {{ border: none; background: none; color: {MUTED}; text-decoration: underline; text-underline-offset: 3px; }}"
            ),
        ],
    )
    # Outcome and location filters affect the game chart, slope chart and menu, not summary counts.
    show_title = Div(text='<div class="label">Outcomes</div>', **widget_style)
    outcome_filter = CheckboxButtonGroup(
        labels=SHOW_LABELS,
        active=[0, 1, 2, 3],
        width=520,
        stylesheets=[PAGE, outcome_filter_css()],
    )
    where_title = Div(text='<div class="label">Location</div>', **widget_style)
    venue_filter = RadioButtonGroup(
        labels=["Home or away", "Home", "Away"], active=0, width=270, **widget_style
    )
    team_summary = Div(width=1240, **widget_style)  # the team summary
    findings = Div(width=1240, **widget_style)  # the answer up front, over every test season
    game_summary = Div(width=1240, **widget_style)  # the clicked game

    # ---------- 1. how often a lead holds ----------
    probability_curves = figure(
        title="Chance a lead holds in the second half",
        width=760,
        height=CURVES_HEIGHT,
        x_range=Range1d(24, 0, bounds="auto"),
        y_range=Range1d(float(spaced(0.36)), float(spaced(0.9995)), bounds="auto"),
        tools="",
    )
    zoomable(probability_curves)
    curve_times = np.round(np.concatenate([np.arange(24, 2, -0.1), np.arange(2, 0.01, -0.02)]), 3)
    curve_sources, curve_labels = [], []
    for ld, color in zip(LEADS, LEAD_COLORS):
        src = ColumnDataSource(
            dict(left=curve_times, y=spaced(hold_probability(ld, curve_times, model)))
        )
        probability_curves.line("left", "y", source=src, line_color=color, line_width=2)
        lab = Label(
            x=LABEL_X,
            y=float(spaced(hold_probability(ld, LABEL_X, model))),
            text=f"up {ld}",
            text_font_size="11px",
            background_fill_color=BG,
            background_fill_alpha=0.85,
            text_color=MUTED,
            y_offset=2,
        )
        probability_curves.add_layout(lab)
        curve_sources.append(src)
        curve_labels.append(lab)
    # Observed hold rates for the reference leads, with 95% intervals.
    dots = dict(
        x=[],
        lead=[],
        p=[],
        lo=[],
        hi=[],
        y=[],
        ylo=[],
        yhi=[],
        marker=[],
        color=[],
        text=[],
    )
    top = float(spaced(0.9995)) - 0.04  # where a 100% real rate is pinned: just inside the top edge
    for ld, color in zip(LEADS, LEAD_COLORS):
        for T in range(24, 1, -4):  # every 4 minutes: every 2 crowded the chart
            n_games, held = observed_counts.get(f"{ld}|{T}", (0, 0))
            if not n_games:
                continue
            lo, hi = wilson(held, n_games)
            dots["x"].append(T)
            dots["lead"].append(ld)
            dots["p"].append(held / n_games)
            dots["lo"].append(lo)
            dots["hi"].append(hi)
            # A 100% hold rate has no finite position on this scale; show it at the top as ▲.
            all_held = held == n_games
            dots["y"].append(top if all_held else float(spaced(held / n_games)))
            dots["marker"].append("triangle" if all_held else "circle")
            dots["ylo"].append(float(spaced(lo)))
            dots["yhi"].append(float(spaced(hi)))
            dots["color"].append(color)
            dots["text"].append(
                f"Up {ld - 1}–{ld + 1} with {T - 1}–{T + 1} min left: {held} of {n_games} held"
            )
    observed_source = ColumnDataSource(dots)
    probability_curves.segment(
        "x",
        "ylo",
        "x",
        "yhi",
        source=observed_source,
        line_color="color",
        line_width=1,
        line_alpha=0.18,
    )
    observed_renderer = probability_curves.scatter(
        "x",
        "y",
        source=observed_source,
        marker="marker",
        size=7,
        fill_color=BG,
        line_color="color",
        line_width=2,
        nonselection_line_alpha=1,
        nonselection_fill_alpha=1,
    )
    probability_curves.add_tools(
        HoverTool(
            renderers=[observed_renderer],
            tooltips=[
                ("Real games", "@text"),
                ("Held", "@p{0.0%}"),
                ("95% range", "@lo{0%} to @hi{0%}"),
                ("Click", "to ask about up @lead with @x min left"),
            ],
        ),
        TapTool(renderers=[observed_renderer]),
    )
    # the model's own uncertainty, for the answer: 100 refits on resampled games (lead_model.bootstrap_fits). Drawn
    # as a band it is about a pixel wide (the refits barely move), so it is given in words instead.
    bootstrap_parameters = [[b["A"], b["beta"], b["kappa"], b["per_elo"]] for b in model["boot"]]
    selected_curve_source = ColumnDataSource(
        dict(left=curve_times, y=spaced(hold_probability(12, curve_times, model)))
    )
    probability_curves.line("left", "y", source=selected_curve_source, line_color=INK, line_width=3)
    point = ColumnDataSource(dict(x=[4.0], y=[float(spaced(hold_probability(12, 4, model)))]))
    probability_curves.scatter(
        "x", "y", source=point, size=12, fill_color=INK, line_color=BG, line_width=2
    )
    # the answer written at the dot, so the chart states its finding (update moves it and keeps it inside the frame)
    point_label = Label(
        x=4.0,
        y=float(spaced(hold_probability(12, 4, model))),
        x_offset=-10,
        y_offset=8,
        text=f"up 12: {np.floor(1000 * hold_probability(12, 4, model) + 1e-9) / 10:.1f}%",
        text_align="right",
        text_font_size="12px",
        text_font_style="bold",
        text_color=INK,
        background_fill_color=BG,
        background_fill_alpha=0.85,
    )
    probability_curves.add_layout(point_label)
    # legend: the curves' meaning, drawn from glyphs with no data (the real ones change with the controls)
    probability_curves.add_layout(
        Legend(
            location="top_center",
            background_fill_alpha=0.95,
            items=[
                LegendItem(
                    label=f"real games ({observed_game_count:,}), 95% range",
                    renderers=[
                        probability_curves.scatter(
                            *OFF,
                            size=7,
                            fill_color=BG,
                            line_color=LEAD_COLORS[2],
                            line_width=2,
                        )
                    ],
                ),
                LegendItem(
                    label="all observed leads held",
                    renderers=[probability_curves.scatter(
                        *OFF, marker="triangle", size=7, fill_color=BG,
                        line_color=LEAD_COLORS[2], line_width=2,
                    )],
                ),
                LegendItem(
                    label="selected probability threshold",
                    renderers=[probability_curves.line(*OFF, line_color=INK, line_dash="dashed")],
                ),
                LegendItem(
                    label=f"close-call threshold ({CLOSE_CALL_THRESHOLD:.0f}%)",
                    renderers=[
                        probability_curves.line(
                            *OFF, line_color=SCARE, line_dash="dotted", line_width=1.8
                        )
                    ],
                ),
            ],
        )
    )
    rest_line = Span(
        location=float(spaced(0.99)),
        dimension="width",
        line_color=INK,
        line_dash="dashed",
        line_width=1,
    )
    probability_curves.add_layout(rest_line)
    probability_curves.add_layout(
        Span(
            location=float(spaced(CLOSE_CALL_THRESHOLD / 100)),
            dimension="width",
            line_color=SCARE,
            line_dash="dotted",
            line_width=1.8,
        )
    )
    ticks = [round(float(spaced(t)), 6) for t in SAFE_TICKS]
    probability_curves.yaxis.ticker = FixedTicker(ticks=ticks)
    probability_curves.yaxis.major_label_overrides = {
        t: f"{100 * v:g}%" for t, v in zip(ticks, SAFE_TICKS)
    }
    probability_curves.yaxis.axis_label = "Chance the lead holds (spaced by the chance of losing)"
    how_to_read(
        probability_curves,
        "Hover for model probabilities. Drag to zoom.",
    )
    hover_probabilities = ColumnDataSource(
        {
            "left": curve_times,
            **{f"up{ld}": hold_probability(ld, curve_times, model) for ld in LEADS},
        }
    )
    hover_line = probability_curves.line(
        "left", "up3", source=hover_probabilities, line_alpha=0
    )  # only its x matters (vline hover)
    probability_curves.add_tools(
        HoverTool(
            renderers=[hover_line],
            mode="vline",
            tooltips=[("Minutes left", "@left{0.0}")]
            + [(f"Up {ld}", f"@up{ld}{{0.0%}}") for ld in reversed(LEADS)]
            + [("Solid curves", "Predictions for each labeled lead"),
               ("Dashed line", "Selected probability threshold"),
               ("Dotted line", "95% reference for close calls")],
        )
    )

    # ---------- 2. the games: margin through regulation, from one team's side ----------
    game_line_source = ColumnDataSource(
        dict(xs=[], ys=[], color=[], label=[], g=[], e=[], alpha=[], width=[])
    )
    rest_markers = ColumnDataSource(
        dict(x=[], y=[], alpha=[], g=[])
    )  # g: each mark's game, for clicking it
    close_call_markers = ColumnDataSource(dict(x=[], y=[], alpha=[], g=[]))
    selected_game = ColumnDataSource(
        dict(g=[initial_game_index])
    )  # index of the clicked game, -1 = none
    rest_frontier = ColumnDataSource(dict(x=[], y=[], top=[]))
    close_call_line = ColumnDataSource(dict(x=[], y=[]))
    game_lines = figure(
        title="The games: score margin from one team's side",
        width=LINES_WIDTH,
        height=LINES_HEIGHT,
        x_range=Range1d(48, 0, bounds="auto"),
        y_range=Range1d(-25, 40, bounds="auto"),
        tools="",
        stylesheets=[CLICKABLE],
    )
    zoomable(game_lines)
    game_line_renderer = game_lines.multi_line(
        "xs",
        "ys",
        source=game_line_source,
        line_color="color",
        line_alpha="alpha",
        line_width="width",
        hover_line_width=4.5,
        hover_line_color=INK,
        hover_line_alpha=1,
        nonselection_line_alpha="alpha",
        selection_line_alpha="alpha",
    )
    game_lines.line(
        "x",
        "y",
        source=rest_frontier,
        line_color=INK,
        line_dash="dashed",
        line_width=1.2,
    )
    game_lines.line(
        "x",
        "y",
        source=close_call_line,
        line_color=SCARE,
        line_dash="dotted",
        line_width=1.8,
    )
    # the line's color already says what happened; the marks are black with a white edge so they stand out on it
    game_lines.scatter(
        "x",
        "y",
        source=rest_markers,
        size=12,
        fill_color=INK,
        fill_alpha="alpha",
        line_color=BG,
        line_width=2,
        line_alpha="alpha",
    )
    game_lines.scatter(
        "x",
        "y",
        source=close_call_markers,
        size=13,
        fill_color=BG,
        fill_alpha="alpha",
        line_color=INK,
        line_width=3,
        line_alpha="alpha",
    )
    game_lines.add_layout(Span(location=0, dimension="width", line_color=INK2, line_width=1))
    # the moment in the picked game that panel 1 is answering (set by clicking the game's line; tags: what was asked)
    ask_line = Span(
        location=-99,
        dimension="height",
        line_color=INK,
        line_dash="dotted",
        line_width=1.5,
        tags=["", -99],
    )
    game_lines.add_layout(ask_line)
    ask_label = Label(
        x=-99,
        y=37,
        text="Selected moment",
        text_font_size="11px",
        text_color=INK,
        x_offset=4,
    )
    game_lines.add_layout(ask_label)
    game_lines.add_layout(
        BoxAnnotation(left=DECISION_MIN, right=0, fill_color=MUTED, fill_alpha=0.1)
    )
    game_lines.add_layout(
        Label(
            x=DECISION_MIN + 0.3,
            y=-23,
            text="last 2 min left out:\nfouls and timeouts\ntake over",
            text_font_size="11px",
            text_color=MUTED,
            text_align="right",
        )
    )
    game_lines.yaxis.axis_label = "Points ahead (+) or behind (−)"
    game_lines.add_tools(
        HoverTool(
            renderers=[game_line_renderer],
            line_policy="interp",
            tooltips=[
                ("Game", "@label"),
                ("Click", "to pick a game, then click again to ask about that moment"),
            ],
        )
    )
    mark = lambda **kw: game_lines.scatter(*OFF, size=11, line_color=BG, fill_color=INK, **kw)
    # Keep legend markers large enough to distinguish their shapes.
    game_lines.add_layout(
        Legend(
            orientation="horizontal",
            location="top_left",
            glyph_width=18,
            glyph_height=18,
            label_height=18,
            spacing=14,
            items=[
                LegendItem(label="threshold reached", renderers=[mark()]),
                LegendItem(
                    label=f"fell under {CLOSE_CALL_THRESHOLD:.0f}% (close call)",
                    renderers=[
                        game_lines.scatter(
                            *OFF, size=10, fill_color=BG, line_color=INK, line_width=2.5
                        )
                    ],
                ),
                LegendItem(
                    label="selected threshold",
                    renderers=[game_lines.line(*OFF, line_color=INK, line_dash="dashed")],
                ),
                LegendItem(
                    label="95% reference",
                    renderers=[
                        game_lines.line(*OFF, line_color=SCARE, line_dash="dotted", line_width=1.8)
                    ],
                ),
            ],
        ),
        "above",
    )
    # added after the legend so it sits right under the title (later "above" items go further from the plot)
    how_to_read(
        game_lines,
        "One line per game, from the team's side: above 0 it's ahead. Click a line to pick it. Click the picked one to ask about that moment. Drag to zoom.",
    )
    # ---------- 2a. under the game chart: how many of the picked game's starters were on the floor ----------
    # no toolbar of its own (it follows the game chart's zoom); the right margin stands in for that chart's toolbar
    strip = figure(
        title="Starters on the floor (shaded: most resting)",
        width=LINES_WIDTH,
        height=STRIP_HEIGHT,
        x_range=game_lines.x_range,
        y_range=Range1d(-0.4, 5.6),
        tools="",
        toolbar_location=None,
    )
    strip.add_layout(BoxAnnotation(bottom=-0.4, top=2.5, fill_color=MUTED, fill_alpha=0.1))
    strip.add_layout(BoxAnnotation(left=DECISION_MIN, right=0, fill_color=MUTED, fill_alpha=0.1))
    starters_src = ColumnDataSource(dict(x=[], y=[]))
    strip.line("x", "y", source=starters_src, line_color=INK, line_width=2)
    strip_note = Label(
        x=24,
        y=2.6,
        text="",
        text_align="center",
        text_font_size="12px",
        text_color=MUTED,
    )
    strip.add_layout(strip_note)
    strip.yaxis.ticker = FixedTicker(ticks=[0, 5])
    strip.yaxis.axis_label = "of 5"
    for p in (game_lines, strip):  # the same left margin, so the two charts' minutes line up
        p.min_border_left = 72
    game_lines.xaxis.visible = False
    for p, ticks in (
        (probability_curves, {24: "Halftime", 18: "18", 12: "4th qtr", 6: "6", 2: "2", 0: "End"}),
        (strip, MIN_TICKS),
        (game_lines, MIN_TICKS),
    ):
        p.xaxis.ticker = FixedTicker(ticks=list(ticks))
        p.xaxis.major_label_overrides = ticks
        p.xaxis.axis_label = "Minutes left"

    # A shared time guide connects the score and starter count under the cursor.
    hover_guides = []
    hover_readouts = []
    for plot in (game_lines, strip):
        guide = Span(
            location=0, dimension="height", line_color=INK2,
            line_width=1, line_dash="dashed", visible=False,
        )
        plot.add_layout(guide)
        hover_guides.append(guide)
        readout = Label(
            x=8, y=8, x_units="screen", y_units="screen",
            text_font_size="12px", text_color=INK,
            background_fill_color=BG, background_fill_alpha=0.95,
            visible=False,
        )
        plot.add_layout(readout)
        hover_readouts.append(readout)
    follow_cursor = CustomJS(
        args=dict(guides=hover_guides, readouts=hover_readouts,
                  time_range=game_lines.x_range, lines=game_line_source,
                  starters=starters_src, focus=selected_game),
        code="""
const x = cb_obj.x;
const inside = Number.isFinite(x) &&
  x >= Math.min(time_range.start, time_range.end) &&
  x <= Math.max(time_range.start, time_range.end);
// Use the displayed whole second for both the guide and the step lookups.
const seconds = Math.round(x * 60);
const time = Math.round(seconds / 60 * 1000) / 1000;
const clock = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
const selected = lines.data.g.indexOf(focus.data.g[0]);
function valueAt(xs, ys) {
  let value = ys[0];
  for (let i = 0; i < xs.length && xs[i] >= time; i++) value = ys[i];
  return value;
}
let text = `${clock} left — select a game for score and starters`;
if (selected >= 0 && starters.data.x.length) {
  const margin = valueAt(lines.data.xs[selected], lines.data.ys[selected]);
  const count = valueAt(starters.data.x, starters.data.y);
  const score = margin > 0 ? `Up ${margin}` : margin < 0 ? `Down ${-margin}` : 'Tied';
  text = `${clock} left   ${score}   ${count} of 5 starters on court`;
}
for (const guide of guides) {
  guide.visible = inside;
  if (inside) guide.location = time;
}
for (const readout of readouts) {
  readout.visible = inside;
  readout.text = text;
}
""",
    )
    hide_cursor = CustomJS(
        args=dict(guides=hover_guides, readouts=hover_readouts),
        code="for (const item of [...guides, ...readouts]) item.visible = false;",
    )
    for plot in (game_lines, strip):
        plot.js_on_event("mousemove", follow_cursor)
        plot.js_on_event("mouseleave", hide_cursor)

    game_line_source.js_on_change("data", hide_cursor)

    # Mark changes between most starters resting and most on court, at the new count.
    substitution_source = ColumnDataSource(dict(x=[], y=[], on=[], marker=[], text=[]))
    substitution_renderer = strip.scatter(
        "x",
        "y",
        source=substitution_source,
        marker="marker",
        size=14,
        fill_color=INK,
        line_color=BG,
        line_width=2,
    )
    strip.add_tools(
        HoverTool(renderers=[substitution_renderer], tooltips=[("Starters", "@text")])
    )
    strip.add_layout(
        Legend(
            orientation="horizontal", location="top_left", spacing=14,
            glyph_width=32, glyph_height=32, label_height=20,
            items=[
                LegendItem(label="Most out (2 or fewer)", renderers=[
                    strip.scatter(*OFF, marker="inverted_triangle", size=14, fill_color=INK, line_color=INK)
                ]),
                LegendItem(label="Most back (3 or more)", renderers=[
                    strip.scatter(*OFF, marker="triangle", size=14, fill_color=INK, line_color=INK)
                ]),
            ],
        ),
        "above",
    )
    # the picked game's marks, labeled where they are (beside the mark, away from the right edge)
    mark_labels = ColumnDataSource(dict(x=[], y=[], text=[], align=[], xo=[], yo=[]))
    game_lines.add_layout(
        LabelSet(
            x="x",
            y="y",
            text="text",
            source=mark_labels,
            text_align="align",
            x_offset="xo",
            y_offset="yo",
            text_baseline="middle",
            text_font_size="11px",
            text_color=INK,
            background_fill_color=BG,
            background_fill_alpha=0.85,
        )
    )

    # ---------- waiting time from the threshold to sustained rest ----------
    waits_src = ColumnDataSource(dict(x=[], held=[], scary=[], lost=[], total=[], gs=[], text=[]))
    waits = figure(
        title="Time until sustained rest",
        min_border_right=16,  # leave room for the last tick label
        width=LINES_WIDTH - SLOPE_WIDTH - GAP,
        height=LOWER_ROW_HEIGHT,
        x_range=Range1d(-1, 25),
        y_range=Range1d(0, 5),
        tools="",
        toolbar_location=None,
        stylesheets=[CLICKABLE],
    )
    waits.vbar_stack(
        ["held", "scary", "lost"],
        x="x",
        width=0.9,
        source=waits_src,
        color=[HELD, SCARE, BLOWN],
        line_color=BG,
    )
    waits_hit = waits.vbar(
        x="x",
        top="total",
        width=0.9,
        source=waits_src,
        fill_alpha=0,
        line_alpha=0,
        nonselection_fill_alpha=0,
        nonselection_line_alpha=0,
    )
    waits_med = Span(
        location=-99,
        dimension="height",
        line_color=INK,
        line_dash="dashed",
        line_width=1,
    )
    waits.add_layout(waits_med)
    waits_note = Label(
        x=-99,
        y=0,
        text="",
        text_font_size="11px",
        text_color=INK2,
        x_offset=4,
        background_fill_color=BG,
        background_fill_alpha=0.85,
    )
    waits.add_layout(waits_note)
    waits.xaxis.axis_label = "Minutes after the probability threshold"
    waits.yaxis.axis_label = "games"
    how_to_read(waits, "One game per entry. Dashed line: median.")
    waits.yaxis.ticker = BasicTicker(min_interval=1, desired_num_ticks=4)  # whole games

    help_height = InlineStyleSheet(css=":host { height: auto !important; flex: 0 0 auto !important; }")
    waits_caption = Div(
        text='<div class="small muted">Wait ends when at most two starters remain through the finish.</div><details class="small muted"><summary>How waits are counted</summary>Negative values mean this happened before the probability threshold. Games without this change are excluded from the median.</details>',
        stylesheets=[PAGE, help_height],
    )

    # ---------- slope chart: threshold lead, least-safe moment and final margin ----------
    slope_source = ColumnDataSource(
        dict(
            xs=[],
            ys=[],
            r=[],
            lo=[],
            f=[],
            rt=[],
            lt=[],
            color=[],
            label=[],
            g=[],
            alpha=[],
            width=[],
        )
    )
    slope = figure(
        title="After the threshold",
        width=SLOPE_WIDTH,
        height=LOWER_ROW_HEIGHT,
        x_range=(-0.2, 2.2),
        y_range=Range1d(-5, 30),
        tools="tap",
        toolbar_location=None,
        stylesheets=[CLICKABLE],
    )  # range fitted in update
    slope_renderer = slope.multi_line(
        "xs",
        "ys",
        source=slope_source,
        line_color="color",
        line_alpha="alpha",
        line_width="width",
        hover_line_width=4.5,
        hover_line_color=INK,
        hover_line_alpha=1,
        nonselection_line_alpha="alpha",
        selection_line_alpha="alpha",
    )
    for x, y in ((0, "r"), (1, "lo"), (2, "f")):
        slope.scatter(
            x=x,
            y=y,
            source=slope_source,
            size=6,
            fill_color="color",
            fill_alpha="alpha",
            line_color=None,
            nonselection_fill_alpha="alpha",
            selection_fill_alpha="alpha",
        )
    slope.add_layout(Span(location=0, dimension="width", line_color=INK2, line_width=1))
    slope.xaxis.ticker = FixedTicker(ticks=[0, 1, 2])
    slope.xaxis.major_label_overrides = {0: "Threshold reached", 1: "Least safe", 2: "Final"}
    slope.yaxis.axis_label = "Points ahead (+) or behind (−)"
    # Align the slope chart's left edge with the game chart.
    slope.min_border_left = 72
    how_to_read(slope, "Least safe: before the final 2 min.")
    slope.xgrid.visible = False
    slope.add_tools(
        HoverTool(
            renderers=[slope_renderer],
            line_policy="interp",
            tooltips=[
                ("Game", "@label"),
                ("Threshold reached", "up @r with @rt left"),
                ("Least safe moment", "@lo{+0} with @lt"),
                ("Meaning", "Lowest win probability after the probability threshold, before the final 2 min"),
                ("Final", "@f{+0}"),
            ],
        )
    )

    # ---------- 4. team bars: comfortable leads and what happened next ----------
    zeros = [0] * len(teams)
    team_bars_source = ColumnDataSource(
        dict(
            team=teams,
            held=zeros,
            scary=zeros,
            lost=zeros,
            a=zeros,
            b=zeros,
            c=zeros,
            n=[""] * len(teams),
            alpha=[0.9] * len(teams),
        )
    )
    team_bars_plot = figure(
        title="Leads reaching the threshold, by team",
        width=BARS_WIDTH,
        height=BOTTOM_HEIGHT,
        y_range=FactorRange(*teams),
        x_range=(0, 10),
        tools="tap",
        toolbar_location=None,
        stylesheets=[CLICKABLE],
    )
    for lo, hi, color in ((None, "a", HELD), ("a", "b", SCARE), ("b", "c", BLOWN)):
        team_bars_plot.hbar(
            y="team",
            left=lo or 0,
            right=hi,
            height=0.7,
            source=team_bars_source,
            fill_color=color,
            fill_alpha="alpha",
            line_color=BG,
            line_width=1,
            nonselection_fill_alpha="alpha",
            hover_fill_alpha=1,
            hover_line_color=INK,
        )
    bar_labels = LabelSet(
        x="c",
        y="team",
        text="n",
        source=team_bars_source,
        x_offset=3,
        y_offset=-6,
        text_font_size="11px",
        text_color=INK2,
    )
    team_bars_plot.add_layout(bar_labels)
    team_bars_plot.xaxis.axis_label = "Games reaching the threshold"
    how_to_read(team_bars_plot, "Click a team to see its games.")
    team_bars_plot.ygrid.visible = False
    team_bars_plot.yaxis.major_label_text_font_size = "11px"  # 30 team names
    team_bars_plot.add_tools(
        HoverTool(
            tooltips=[
                ("Team", "@team"),
                ("Held", "@held"),
                ("Close call, still won", "@scary"),
                ("Lost", "@lost"),
                ("Click", "to show this team (again: all teams)"),
            ]
        )
    )

    # ---------- shared update callback ----------
    update = CustomJS(
        args=dict(
            G=game_data,
            season=season_select,
            MATCHUP_KEYS=MATCHUP_KEYS,
            COST_SEASON=cost_season,
            Z=threshold_z_scores,
            M=model_parameters,
            SCARY=CLOSE_CALL_THRESHOLD,
            DECISION=DECISION_MIN,
            ALL=ALL_TEAMS,
            HELD=HELD,
            SCARE=SCARE,
            BLOWN=BLOWN,
            NEVER=FADED,
            OBS=observed_counts,
            sel=team_select,
            level=rest_threshold,
            lead=lead_input,
            left=time_input,
            matchup=matchup_control,
            answer=answer,
            input_notice=input_notice,
            caption=team_summary,
            curve_srcs=curve_sources,
            curve_labels=curve_labels,
            LEADS=LEADS,
            wide=hover_probabilities,
            yours=selected_curve_source,
            point=point,
            point_label=point_label,
            BOOT=bootstrap_parameters,
            mark_labels=mark_labels,
            waits_src=waits_src,
            waits_x=waits.x_range,
            waits_y=waits.y_range,
            waits_med=waits_med,
            waits_note=waits_note,
            waits_caption=waits_caption,
            RESTS=REST_THRESHOLDS,
            EDGES=[e for _, e in MATCHUPS],
            matchup_note=matchup_note,
            ask_label=ask_label,
            scenario_context=scenario_context,
            starters_src=starters_src,
            strip_note=strip_note,
            strip=strip,
            game_axis=game_lines.xaxis[0],
            rest_line=rest_line,
            ask_line=ask_line,
            on=game_line_source,
            firsts=rest_markers,
            dangers=close_call_markers,
            frontier=rest_frontier,
            scary_line=close_call_line,
            slope_on=slope_source,
            bars=team_bars_source,
            bar_y=team_bars_plot.y_range,
            bar_x=team_bars_plot.x_range,
            focus=selected_game,
            card=game_summary,
            findings=findings,
            TESTS=TEST_SEASONS,
            Y_CAP=Y_CAP,
            LABEL_X=LABEL_X,
            slope_y=slope.y_range,
            bar_labels=bar_labels,
            level_note=level_note,
            turns=substitution_source,
            GLOSSARY=GLOSSARY,
            show=outcome_filter,
            where=venue_filter,
            game_pick=game_pick,
            clear_game=clear_game,
        ),
        code="""
const team = sel.value,
  all = team === ALL,
  pct = RESTS[level.active];
const GS = G.season.map((_, g) => g).filter((g) => G.season[g] === season.value),
  n = GS.length; // the season's games
const z = Z[pct.toFixed(1)],
  zs = Z[SCARY.toFixed(1)],
  whatIf = matchup.active == null ? matchup.tags[0] : EDGES[matchup.active];
const erf = (x) => {
  const s = Math.sign(x);
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  return (
    s *
    (1 -
      ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
        t *
        Math.exp(-x * x))
  );
};
const Phi = (x) => 0.5 * (1 + erf(x / Math.SQRT2));
// lead_model.py: a lead L with t minutes left, for a team e Elo better, ends regulation ~ Normal(mean, sd)
const sd = (t) => M.A * Math.pow(t, M.beta),
  drift = (e, t) => (M.per_elo * e * t) / 48;
const hold = (L, t, e) => Phi((L * (1 - M.kappa * Math.sqrt(t)) + drift(e, t)) / sd(t));
// the curves assume the usual lineups (both coaches tend to bench together); if you sit all five and they keep all
// five on, every minute costs per_starter x 5 against that usual gap (lead_model.fit_starter_effect)
// a team whose starters are worth more than the league's, beyond chance (lead_model.team_starter_values), uses its own value
const perStarter = M.team_per_starter[team] ?? M.per_starter;
const holdSit = (L, t, e) =>
  Phi(
    (L * (1 - M.kappa * Math.sqrt(t)) + drift(e, t) + perStarter * (-5 - M.usual_starter_gap) * t) /
      sd(t),
  );
const needAt = (zz, t, e) => (zz * sd(t) - drift(e, t)) / (1 - M.kappa * Math.sqrt(t)); // lead where hold() = Phi(zz)
const need = (t, e) => needAt(z, Math.max(t, 1e-6), e),
  needScary = (t, e) => needAt(zs, Math.max(t, 1e-6), e);
const clock = (t) => {
  const secs = Math.round(t * 60);
  return `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, '0')}`;
};
// rounded down, so "96.0%" is never under 96%; above 99.9% it says so rather than claiming 100%
const pc = (p) =>
  p >= 0.999 && p < 1 ? '>99.9%' : (Math.floor(1000 * p + 1e-9) / 10).toFixed(1) + '%';
// the chance of losing as a frequency: 1 in 100, 1 in 29 (two significant figures past 10)
const oneIn = (p) =>
  1 - p < 0.001
    ? 'fewer than 1 in 1,000'
    : `about 1 in ${(1 / (1 - p) < 10 ? Math.round(1 / (1 - p)) : Number((1 / (1 - p)).toPrecision(2))).toLocaleString('en-US')}`;
const sideOf = (g, t) => (G.home[g] === t ? 1 : -1),
  won = (g, s) => s * G.final[g] > 0;

// ---- the answer and the curves (the what-if matchup) ----
// clean the number boxes: a whole lead of 1-40 points, 0.5-24 minutes by half minutes, an empty box keeps its last
// value. Writing a cleaned value back re-runs this callback, which then draws.
const clean = (box, lo, hi, step) =>
  box.value == null || isNaN(box.value)
    ? box.tags[0]
    : Math.min(hi, Math.max(lo, Math.round(box.value / step) * step));
// the time box takes a clock ("4:30") or minutes ("4.5"); anything else keeps the last good time
const asMinutes = (v) => {
  const t = String(v ?? '').trim(),
    m = t.match(/^(\\d{1,2}):(\\d{1,2})$/);
  return m ? Number(m[1]) + Number(m[2]) / 60 : /^\\d*\\.?\\d+$/.test(t) ? Number(t) : NaN;
};
const L = clean(lead, 1, 40, 1),
  t0 = asMinutes(left.value);
const T = isNaN(t0) ? left.tags[0] : Math.min(24, Math.max(0.5, Math.round(t0 * 2) / 2));
if (L !== lead.value) {
  input_notice.text = `<div class="small muted" role="status">Points ahead adjusted to ${L}. Use whole numbers from 1 to 40.</div>`;
  input_notice.tags = [L, clock(T)];
  input_notice.visible = true;
  lead.value = L;
  return;
}
if (clock(T) !== left.value) {
  const seconds = String(left.value).split(':')[1];
  const message = isNaN(t0)
    ? `Time not recognized. Kept ${clock(T)}. Use M:SS or decimal minutes.`
    : seconds !== undefined && Number(seconds) >= 60
      ? `Seconds must be 00–59. Your entry was interpreted as ${clock(T)}.`
      : t0 < 0.5 || t0 > 24
        ? `Time is limited to 0:30–24:00. Using ${clock(T)}.`
        : T !== t0 ? `Time rounded to ${clock(T)} (30-second steps).` : '';
  input_notice.text = message ? `<div class="small muted" role="status">${message}</div>` : '';
  input_notice.tags = [L, clock(T)];
  input_notice.visible = Boolean(message);
  left.value = clock(T);
  return;
}
if (input_notice.tags.length && (L !== input_notice.tags[0] || clock(T) !== input_notice.tags[1])) {
  input_notice.text = '';
  input_notice.visible = false;
  input_notice.tags = [];
}
lead.tags = [L];
left.tags = [T];
{
  const fav = drift(whatIf, 48),
    pts = Math.round(Math.abs(fav));
  matchup_note.text =
    `<div class="small muted">${
      pts < 1
        ? 'Neither team favored'
        : `${fav > 0 ? 'Favored' : 'The underdog'} by about ${pts} point${pts === 1 ? '' : 's'} over a full game`
    }` + `${matchup.active == null ? `, as in ${matchup.tags[1] || 'a linked game'}` : ''}</div>`;
}
// Wilson 95% range for k of m (stays inside 0-100% near the ends)
const wil = (k, m) => {
  const q = k / m,
    z2 = 3.8416,
    d = 1 + z2 / m,
    c = (q + z2 / (2 * m)) / d,
    h = (Math.sqrt((q * (1 - q)) / m + z2 / (4 * m * m)) * 1.96) / d;
  return [c - h, c + h];
};
// the same model refit on resampled games: its 95% range (numpy's default quantile, linear between values)
const holdB = (b, L, t, e) =>
  Phi((L * (1 - b[2] * Math.sqrt(t)) + (b[3] * e * t) / 48) / (b[0] * Math.pow(t, b[1])));
const quant = (v, q) => {
  const i = q * (v.length - 1),
    f = Math.floor(i);
  return f + 1 < v.length ? v[f] + (i - f) * (v[f + 1] - v[f]) : v[f];
};
const range95 = (L, t, e) => {
  const v = BOOT.map((b) => holdB(b, L, t, e)).sort((a, b) => a - b);
  return [quant(v, 0.025), quant(v, 0.975)];
};
const p = hold(L, T, whatIf),
  needNow = Math.max(1, Math.ceil(need(T, whatIf)));
// Classify the predicted probability against two descriptive thresholds.
const [verdict, tone, why] =
  T < DECISION
    ? ['Last 2 minutes', 'muted', 'Possession and fouling matter more than the lead here.']
    : p >= pct / 100
      ? [
          'Probability threshold reached',
          'held',
          `The main model reaches ${pct}% at ${needNow}+ points with ${clock(T)} left.`,
        ]
      : p < SCARY / 100
        ? [
            'Below the 95% reference',
            'lost',
            `Below ${SCARY}% predicted chance. The ${pct}% threshold is ${needNow}+ points here.`,
          ]
        : [
            `Below threshold: ${needNow} points needed`,
            'lost',
            `The main model reaches ${pct}% at ${needNow}+ points with ${clock(T)} left.`,
          ];
const Lr = L,
  Tr = T,
  ob = OBS[`${L}|${T}`];
const lo = Math.max(1, Math.ceil(T - 1)),
  hi = Math.floor(T + 1); // the whole-minute checkpoints in the window
const pct1 = (q) => (100 * q).toFixed(1) + '%',
  [rLo, rHi] = ob ? wil(ob[1], ob[0]) : [0, 0];
const window_ = `leads of ${Math.max(1, Lr - 1)}–${Lr + 1} with ${lo === hi ? clock(lo) : clock(lo) + '–' + clock(hi)} left`;
const [mLo, mHi] = range95(L, T, whatIf);
// Show the threshold comparison and probability, with intervals in expandable details.
const af = (k, v) =>
  `<div class="fact"><span class="k">${k}</span><span class="v">${v}</span></div>`;
// Div text is rebuilt on updates; retain the reader's expanded explanations.
const detailsOpen = (model) => Bokeh.index.find_one(model)?.el.shadowRoot?.querySelector('details')?.open ? ' open' : '';
const sitP = holdSit(L, T, whatIf);
answer.text =
  `<div class="answer"><div><div class="verdict ${tone}">${verdict}</div><div class="small muted">${why}</div></div>` +
  `<div><div class="ink">${T < DECISION ? "Model estimate only (last 2 minutes)" : "Chance the lead holds under observed coaching"}</div><div class="big">${pc(p)}</div></div>` +
  (T >= DECISION
    ? `<div class="comparison"><b>${pc(sitP)}</b><span class="small muted">Lineup-adjusted scenario (observational)</span></div>`
    : '') +
  `<details${detailsOpen(answer)} class="small muted"><summary>Evidence and assumptions</summary>` +
  `<div class="small muted">The main model reflects observed coaching, including starters returning. Crossing the threshold does not show that starters can safely stay out.</div>` +
  `<div>The lineup scenario assumes all five starters sit while the opponent keeps theirs. Its fitted adjustment is observational, not an estimate from a controlled substitution policy.</div>` +
  `<div>Lost in ${oneIn(p)} games like this under the main model.</div><div class="facts">` +
  af('Real games like it', ob ? `<b>${ob[1]} of ${ob[0]}</b> held` : 'none yet') +
  `</div>` +
  (mLo < 0.999 ? `The model's 95% range: ${pc(mLo)}–${pc(mHi)}. ` : '') +
  (ob ? `Real games: ${window_}, 95% range ${pct1(rLo)}–${pct1(rHi)}. ` : '') +
  (T >= DECISION ? `Sitting all five: lost in ${oneIn(sitP)}. ` : '') +
  (team in M.team_per_starter
    ? `${team}'s starters are worth ${(perStarter / M.per_starter).toFixed(1)}× the league's, so its own value is used.`
    : '') +
  `</details></div>`;
const tl = wide.data.left,
  W = { left: tl };
const sp = (q) => Math.min(-Math.log10(Math.max(1 - q, 1e-12)), Y_CAP); // spaced(): the curves' y-axis
LEADS.forEach((ld, i) => {
  const ps = tl.map((t) => hold(ld, t, whatIf));
  curve_srcs[i].data = { left: tl, y: ps.map(sp) };
  W['up' + ld] = ps;
  curve_labels[i].y = sp(hold(ld, LABEL_X, whatIf));
});
wide.data = W;
yours.data = { left: tl, y: tl.map((t) => sp(hold(L, t, whatIf))) };
point.data = { x: [T], y: [sp(p)] };
const right = T < 8; // near the right edge the label goes on the dot's left, else on its right
point_label.setv({
  x: T,
  y: sp(p),
  text: `up ${L}: ${pc(p)}`,
  text_align: right ? 'right' : 'left',
  x_offset: right ? -10 : 10,
  y_offset: sp(p) > Y_CAP - 0.3 ? -22 : 8,
});
rest_line.location = sp(pct / 100);

// ---- first threshold crossing and later close calls for each side ----
function walk(g, s) {
  const xs = G.xs[g],
    ys = G.ys[g],
    e = s * G.edge[g];
  let first = -1;
  for (let i = 0; i < xs.length; i++) {
    if (xs[i] <= 0) break;
    const ld = s * ys[i];
    if (first < 0) {
      if (xs[i] >= DECISION && ld > 0 && ld >= need(xs[i], e)) first = i;
    } else if (xs[i] <= DECISION)
      break; // like the rest decision, a scare only counts before the last two minutes
    else if (ld < needScary(xs[i], e)) return { first, danger: i, scary: true };
  }
  return { first, danger: -1, scary: false };
}
// Find the lowest hold probability after the threshold, excluding the final two minutes.
function closest(g, s, w) {
  const xs = G.xs[g],
    ys = G.ys[g],
    e = s * G.edge[g];
  let lo = w.first,
    pr = hold(s * ys[w.first], xs[w.first], e);
  for (let i = w.first + 1; i < xs.length && xs[i] > DECISION; i++) {
    const q = hold(s * ys[i], xs[i], e);
    if (q < pr) {
      pr = q;
      lo = i;
    }
  }
  return { i: lo, p: pr, t: xs[lo], lead: s * ys[lo] };
}
const outcome = (g, s, w) => (w.first < 0 ? NEVER : !won(g, s) ? BLOWN : w.scary ? SCARE : HELD);
const full = []; // every game (all teams: every rested lead) from the leading side, before the filters
let comfy = 0,
  scary = 0,
  lost = 0;
for (const g of GS) {
  // the same three outcomes as the team bars: held, a close call but still won, lost
  const tally = (g, s, w) => {
    comfy++;
    if (!won(g, s)) lost++;
    else if (w.scary) scary++;
  };
  if (all) {
    for (const s of [1, -1]) {
      const w = walk(g, s);
      if (w.first < 0) continue;
      tally(g, s, w);
      full.push([g, s, w, outcome(g, s, w)]);
    }
  } else if (G.home[g] === team || G.away[g] === team) {
    const s = sideOf(g, team),
      w = walk(g, s);
    if (w.first >= 0) tally(g, s, w);
    full.push([g, s, w, outcome(g, s, w)]);
  }
}
// the filters: which outcomes, and where that side played. They thin the game chart, slope chart and game menu;
// the counts, team bars and waits histogram stay whole
const SHOWN = new Set(show.active.map((i) => [HELD, SCARE, BLOWN, NEVER][i]));
const pick = full.filter(
  ([g, s, w, color]) => SHOWN.has(color) && (where.active === 0 || s > 0 === (where.active === 1)),
);
let fg = focus.data.g[0];
const retained = fg >= 0 && full.some(q => q[0] === fg) && !pick.some(q => q[0] === fg);
// a picked game stays drawn even if the filters would hide it (picked from the histogram or a link)
if (fg >= 0 && !pick.some((q) => q[0] === fg)) {
  const q = full.find((q) => q[0] === fg);
  if (q) pick.push(q);
}
if (fg >= 0 && !pick.some((q) => q[0] === fg)) {
  fg = -1;
  focus.data = { g: [-1] };
}
const rank = (q) => (q[0] === fg ? 4 : { [NEVER]: 0, [HELD]: 1, [SCARE]: 2, [BLOWN]: 3 }[q[3]]);
pick.sort((a, b) => rank(a) - rank(b)); // gray first, focused game last so it draws on top
const O = { xs: [], ys: [], color: [], label: [], g: [], e: [], alpha: [], width: [] },
  F = { x: [], y: [], alpha: [], g: [] };
const D = { x: [], y: [], alpha: [], g: [] };
const S = {
  xs: [],
  ys: [],
  r: [],
  lo: [],
  f: [],
  rt: [],
  lt: [],
  color: [],
  label: [],
  g: [],
  alpha: [],
  width: [],
};
for (const [g, s, w, color] of pick) {
  const a = fg < 0 ? (all ? 0.18 : color === NEVER ? 0.5 : 0.85) : g === fg ? 1 : 0.1,
    wd = g === fg ? 3.5 : all ? 0.8 : 1.6;
  const ma = g === fg || (!all && fg < 0) ? 1 : 0; // League-wide markers appear only for the selected game.
  O.xs.push(G.xs[g]);
  O.ys.push(G.ys[g].map((v) => s * v));
  O.color.push(color);
  O.label.push(G.label[g]);
  O.g.push(g);
  O.e.push(s * G.edge[g]);
  O.alpha.push(a);
  O.width.push(wd);
  if (w.first >= 0) {
    F.x.push(G.xs[g][w.first]);
    F.y.push(s * G.ys[g][w.first]);
    F.alpha.push(ma);
    F.g.push(g);
  }
  if (w.danger >= 0) {
    D.x.push(G.xs[g][w.danger]);
    D.y.push(s * G.ys[g][w.danger]);
    D.alpha.push(ma);
    D.g.push(g);
  }
  if (w.first < 0) continue; // the slope chart only has games where the starters could rest
  const c = closest(g, s, w),
    r = s * G.ys[g][w.first],
    fin = s * G.final[g];
  S.xs.push([0, 1, 2]);
  S.ys.push([r, c.lead, fin]);
  S.r.push(r);
  S.lo.push(c.lead);
  S.f.push(fin);
  S.rt.push(clock(G.xs[g][w.first]));
  S.lt.push(`${clock(c.t)} left, ${pc(c.p)} safe`);
  S.color.push(color);
  S.label.push(G.label[g]);
  S.g.push(g);
  S.alpha.push(a);
  S.width.push(wd);
}
on.data = O;
on.tags = [];
firsts.data = F;
dangers.data = D;
slope_on.data = S;
ask_line.location =
  fg >= 0 && ask_line.tags[0] === `${fg}|${L}|${T}|${whatIf}` ? ask_line.tags[1] : -99;
ask_label.x = ask_line.location;
scenario_context.text = ask_line.location >= 0
  ? `<div class="small muted">From ${G.label[fg].split(':')[0]}, ${clock(T)} remaining<br>Time rounded to the nearest 30 seconds.</div>`
  : `<div class="small muted">${matchup.active == null
      ? `Custom scenario with matchup from ${matchup.tags[1] || 'the linked game'}`
      : ''}</div>`;
scenario_context.visible = ask_line.location >= 0 || matchup.active == null;
// the slope chart's range fits the games it shows, always including zero (a lost lead crosses it)
const sv = S.ys.flat();
slope_y.start = Math.min(0, ...sv) - 3;
slope_y.end = Math.max(10, ...sv) + 3;

// Draw threshold lines only for the selected game because they depend on its matchup.
const lineEdge = fg >= 0 ? pick.find((q) => q[0] === fg)[1] * G.edge[fg] : null;
const fx = [],
  fy = [],
  ft = [],
  bx = [],
  by = [];
for (let t = 48; lineEdge !== null && t >= 0; t -= 0.25) {
  bx.push(t);
  by.push(needScary(t, lineEdge));
  if (t >= DECISION) {
    fx.push(t);
    fy.push(need(t, lineEdge));
    ft.push(60);
  }
}
frontier.data = { x: fx, y: fy, top: ft };
scary_line.data = { x: bx, y: by };

// ---- team bars ----
const B = {};
for (const t of bars.data.team) B[t] = { held: 0, scary: 0, lost: 0 };
for (const g of GS)
  for (const s of [1, -1]) {
    const w = walk(g, s);
    if (w.first < 0) continue;
    const me = s > 0 ? G.home[g] : G.away[g];
    if (!won(g, s)) B[me].lost++;
    else if (w.scary) B[me].scary++;
    else B[me].held++;
  }
const bad = (t) => B[t].scary + B[t].lost,
  tot = (t) => bad(t) + B[t].held;
const TT = [...bars.data.team].sort(
  (a, b) => bad(a) - bad(b) || tot(a) - tot(b) || (a < b ? 1 : -1),
); // most scares on top
bars.data = {
  team: TT,
  held: TT.map((t) => B[t].held),
  scary: TT.map((t) => B[t].scary),
  lost: TT.map((t) => B[t].lost),
  a: TT.map((t) => B[t].held),
  b: TT.map((t) => B[t].held + B[t].scary),
  c: TT.map(tot),
  n: TT.map((t) => (tot(t) ? String(tot(t)) : '')),
  alpha: TT.map((t) => (all || t === team ? 0.9 : 0.25)),
};
bar_y.factors = TT;
bar_x.end = Math.max(...TT.map(tot)) * 1.12 + 1.5; // room for the totals, two digits included

// ---- observed outcomes at the selected threshold over one test season ----
// Count crossings at every scoring change using each game’s matchup.
let cN = 0,
  cLost = 0;
for (let g = 0; g < G.home.length; g++)
  if (G.season[g] === COST_SEASON)
    for (const s of [1, -1]) {
      const w = walk(g, s);
      if (w.first < 0) continue;
      cN++;
      if (!won(g, s)) cLost++;
    }
const nTeams = new Set(G.home.filter((_, g) => G.season[g] === COST_SEASON)).size;
level_note.text =
  `<div class="small">${COST_SEASON}: <b>${Math.round(cN / nTeams)}</b> threshold crossings per team, ` +
  `<b class="${cLost ? 'lost' : 'muted'}">${cLost}</b> lost league-wide</div>`;

// ---- when the coach actually pulled the starters: the last time 3+ were on the floor in regulation ----
function pulled(g, s) {
  // minutes left when at most 2 starters stayed on the floor to the end; null if 3+ were on at the end
  const st = G.st[g][s > 0 ? 'home' : 'away'],
    xs = st[0],
    ys = st[1];
  let last = -1;
  for (let i = 0; i < xs.length - 1; i++) if (ys[i] >= 3) last = i;
  return last < 0 ? xs[0] : last + 1 < xs.length - 1 ? xs[last + 1] : null;
}
const gaps = [],
  waitsBy = [];
let pulledN = 0,
  restN = 0;
const binOf = { [HELD]: 'held', [SCARE]: 'scary', [BLOWN]: 'lost' }; // outcome() colors -> histogram stacks
const addWait = (g, s, w) => {
  restN++;
  const pt = pulled(g, s);
  if (pt !== null) {
    pulledN++;
    gaps.push(G.xs[g][w.first] - pt);
    waitsBy.push([G.xs[g][w.first] - pt, binOf[outcome(g, s, w)], g]);
  }
};
for (const [g, s, w] of all ? [] : full) if (w.first >= 0) addWait(g, s, w);
if (all)
  for (const g of GS)
    for (const s of [1, -1]) {
      const w = walk(g, s);
      if (w.first >= 0) addWait(g, s, w);
    }
gaps.sort((a, b) => a - b);
const med = gaps.length
  ? gaps.length % 2
    ? gaps[(gaps.length - 1) / 2]
    : (gaps[gaps.length / 2 - 1] + gaps[gaps.length / 2]) / 2
  : 0;

// the waits histogram: 1-minute bins, stacked by what happened, the median dashed
{
  const bins = {};
  for (const [v, o, g] of waitsBy) {
    const b = Math.floor(v) + 0.5;
    const k = (bins[b] ??= { held: 0, scary: 0, lost: 0, gs: [] });
    k[o]++;
    k.gs.push(g);
  }
  const xs = Object.keys(bins)
    .map(Number)
    .sort((a, b) => a - b);
  const n = (b) => bins[b].held + bins[b].scary + bins[b].lost;
  waits_src.data = {
    x: xs,
    held: xs.map((b) => bins[b].held),
    scary: xs.map((b) => bins[b].scary),
    lost: xs.map((b) => bins[b].lost),
    total: xs.map(n),
    gs: xs.map((b) => bins[b].gs.sort((a, c) => a - c)),
    text: xs.map(
      (b) =>
        `waited ${b - 0.5} to ${b + 0.5} min: ${n(b)} game${n(b) > 1 ? 's' : ''} (click to pick)`,
    ),
  };
  const top = Math.max(1, ...xs.map((b) => bins[b].held + bins[b].scary + bins[b].lost));
  waits_x.start = Math.min(-1, Math.floor(Math.min(0, ...gaps)));
  waits_x.end = Math.max(25, Math.ceil(Math.max(0, ...gaps)) + 1);
  waits_y.end = top * 1.25 + 0.5;
  waits_med.location = gaps.length ? med : -99;
  const never = restN - pulledN;
  waits_note.setv({
    x: gaps.length ? med : -99,
    y: top * 1.1,
    text: gaps.length ? `median ${med.toFixed(1)} min` : '',
    text_align: med > (waits_x.start + waits_x.end) / 2 ? 'right' : 'left',
    x_offset: med > (waits_x.start + waits_x.end) / 2 ? -4 : 4,
  });
  waits_caption.text =
    (never ? `<div class="small muted">${never} not rested to the finish (excluded from the median).</div>` : '') +
    '<div class="small muted">Wait ends when at most two starters remain through the finish.</div>' +
    `<details${detailsOpen(waits_caption)} class="small muted"><summary>How waits are counted</summary>Negative values mean this happened before the probability threshold. Games without this change are excluded from the median.</details>`;
}

// Crossings between at most two starters and at least three, optionally after a threshold.
function turnsAfter(g, s, could = Infinity) {
  const st = G.st[g][s > 0 ? 'home' : 'away'],
    ev = [];
  for (let i = 1; i < st[0].length - 1; i++) {
    const x = st[0][i],
      before = st[1][i - 1],
      now = st[1][i];
    if (x >= could) continue;
    if (before >= 3 && now <= 2) ev.push({ x, out: true, on: now });
    if (before <= 2 && now >= 3) ev.push({ x, out: false, on: now });
  }
  return ev;
}
function marginAt(g, s, x) {
  const xs = G.xs[g],
    ys = G.ys[g];
  let v = ys[0];
  for (let i = 0; i < xs.length && xs[i] >= x; i++) v = ys[i];
  return s * v;
}
const leadText = (L) => (L > 0 ? 'up ' + L : L < 0 ? 'down ' + -L : 'tied');

// ---- caption ----
// Labels sit above their values in the game summary.
const fact = (k, v, cls = '') =>
  `<div class="fact ${cls}"><span class="k">${k}</span><span class="v">${v}</span></div>`;
caption.text =
  `<div class="cap">${all ? 'All teams' : `<b>${team}</b>`}: <b>${comfy}</b> of ${all ? n : full.length} games had a lead ` +
  `that reached the selected threshold` +
  (pulledN
    ? `. ${all ? 'Coaches' : 'The coach'} rested most starters to the finish a median <b>${med.toFixed(1)} min</b> later`
    : '') +
  `.</div>`;
{
  const lab = [
    `Held ${comfy - scary - lost}`,
    `Close call ${scary}`,
    `Lost ${lost}`,
    `Below threshold ${(all ? n : full.length) - comfy}`,
  ];
  if (show.labels.join() !== lab.join()) show.labels = lab;
}
let text = '';
if (fg >= 0) {
  const [g, s, w] = pick.find((q) => q[0] === fg),
    xs = G.xs[g],
    ys = G.ys[g],
    e = s * G.edge[g];
  const me = s > 0 ? G.home[g] : G.away[g],
    [match, score] = G.label[g].split(': ');
  const dr = drift(e, 48),
    strength =
      Math.abs(e) < 25
        ? 'even matchup'
        : `${me} expected to ${dr > 0 ? 'win' : 'lose'} by ${Math.max(1, Math.round(Math.abs(dr)))}`;
  // Match the summary labels to the chart markers.
  text =
    fact(match, score, 'head') +
    fact(`Going in <abbr title="${GLOSSARY.matchup}">(matchup)</abbr>`, strength);
  if (w.first >= 0) {
    const c = closest(g, s, w);
    text +=
      fact('● Threshold reached', `<b>${clock(xs[w.first])}</b> left, up ${s * ys[w.first]}`) +
      (c.i > w.first
        ? fact(
            c.p < SCARY / 100
              ? `Least safe moment (<abbr title="${GLOSSARY['close call']}">close call</abbr>)`
              : 'Least safe moment',
            `<b>${clock(c.t)}</b> left, ${leadText(c.lead)} <span class="muted">(${pc(c.p)} safe)</span>`,
          )
        : '');
  } else text += fact('Threshold reached', 'never reached threshold');
  text += fact('Final', `${me} ${won(g, s) ? 'won' : 'lost'} by ${Math.abs(G.final[g])}`);
}
// the picked game as one plain sentence: what the lead did, and what the coach did
let story = '';
if (fg >= 0) {
  const [g, s, w] = pick.find((q) => q[0] === fg),
    me = s > 0 ? G.home[g] : G.away[g];
  if (w.first >= 0) {
    const c = closest(g, s, w),
      ev = turnsAfter(g, s, G.xs[g][w.first]).filter((t) => t.x >= 1 / 60);
    // List substitutions in time order around the least-safe moment.
    const when = (x) =>
      [36, 24, 12].includes(x)
        ? `at the start of the ${{ 36: '2nd', 24: '3rd', 12: '4th' }[x]} quarter`
        : `with ${clock(x)} left`;
    const say = (es) =>
      es.map((t) => (t.out ? `sat them ${when(t.x)}` : `sent ${t.on} back in ${when(t.x)}`));
    const join = (m) =>
      m.length < 2 ? m.join('') : m.slice(0, -1).join(', ') + ' and ' + m.at(-1);
    const low = c.i > w.first ? c.t : -1,
      before = say(ev.filter((t) => t.x > low)),
      after = say(ev.filter((t) => t.x <= low));
    const st = G.st[g][s > 0 ? 'home' : 'away'];
    const onAt = (x) => {
      let v = st[1][0];
      for (let i = 0; i < st[0].length && st[0][i] >= x; i++) v = st[1][i];
      return v;
    };
    const on0 = onAt(G.xs[g][w.first]);
    const parts = [];
    if (on0 <= 2)
      parts.push(
        `Most starters were already resting when the threshold was reached (${on0} of 5 on the floor).`,
      );
    if (before.length) parts.push(`The coach ${join(before)}.`);
    if (low >= 0 && c.p < SCARY / 100)
      parts.push(`At the least safe moment, ${onAt(c.t)} of 5 starters were on the floor.`);
    if (after.length)
      parts.push(`${before.length ? 'Then the coach' : 'The coach'} ${join(after)}.`);
    if (!parts.length) parts.push('The coach kept most of the starters on to the end.');
    story = parts.join(' ');
  }
}
card.text = fg < 0
  ? `<div class="game"><div class="small muted">Select a game from the menu or click a line to inspect it.</div></div>`
  : `<div class="game"><details${detailsOpen(card)}><summary><b>${G.label[fg]}</b> <span class="small muted">Game details</span></summary><div class="facts">${text}</div>${story ? `<div class="story">${story}</div>` : ''}</details>` +
    (retained ? '<div class="small muted">Selected game stays visible outside the current outcome or location filters.</div>' : '') +
    (ask_line.location >= 0 ? `<div class="small">Selected moment: ${clock(T)} left, up ${L}. <b>${pc(p)}</b> ${T < DECISION ? "model estimate only" : "chance the lead holds"}. <span class="${tone}">${verdict}.</span></div>` : '') +
    '</div>';

// ---- findings: every test season, the chosen threshold ----
let fN = 0,
  fLost = 0;
const fGaps = [];
for (let g = 0; g < G.home.length; g++)
  for (const s of [1, -1]) {
    const w = walk(g, s);
    if (w.first < 0) continue;
    fN++;
    if (!won(g, s)) fLost++;
    const pt = pulled(g, s);
    if (pt !== null) fGaps.push(G.xs[g][w.first] - pt);
  }
fGaps.sort((a, b) => a - b);
const fMed = fGaps.length
  ? fGaps[Math.floor((fGaps.length - 1) / 2)] / 2 + fGaps[Math.ceil((fGaps.length - 1) / 2)] / 2
  : 0;
const [fLo, fHi] = fN ? wil(fLost, fN) : [0, 0];
const tile = (num, text, sub) =>
  `<div class="tile"><span class="num">${num}</span><div class="ink">${text}</div><div class="small muted">${sub}</div></div>`;
findings.text =
  `<div class="tiles">` +
  tile(
    `${Math.ceil(needAt(z, 5, 0))} points`,
    'lead needed with 5:00 left',
    `model estimate for ${pct}% hold probability, even matchup`,
  ) +
  tile(
    `${fLost} of ${fN.toLocaleString()}`,
    'lost after reaching the probability threshold',
    `${TESTS.join(' and ')}, 95% range ${(100 * fLo).toFixed(1)}–${(100 * fHi).toFixed(1)}%`,
  ) +
  tile(
    `${fMed.toFixed(1)} min`,
    'until most starters rested to the finish',
    `median, after reaching ${pct}% predicted probability`,
  ) +
  `</div>`;

// ---- the selected game: majority crossings on the starter strip, score labels above ----
if (fg >= 0) {
  const [g, s, w] = pick.find((q) => q[0] === fg);
  const ev = turnsAfter(g, s),
    e = s * G.edge[g];
  turns.data = {
    x: ev.map((t) => t.x),
    y: ev.map((t) => t.on),
    on: ev.map((t) => t.on),
    marker: ev.map((t) => (t.out ? 'inverted_triangle' : 'triangle')),
    text: ev.map(
      (t) =>
        `${t.out ? 'most starters out' : 'most starters back'}: ${t.on} of 5 on with ${clock(t.x)} left, ` +
        `${leadText(marginAt(g, s, t.x))} (${pc(hold(marginAt(g, s, t.x), t.x, e))} safe)`,
    ),
  };
  // labels at the picked game's marks: beside each, on the left near the right edge; above or below the line
  const ML = { x: [], y: [], text: [], align: [], xo: [], yo: [] };
  const lab = (x, y, text, up) => {
    const r = x < 12;
    ML.x.push(x);
    ML.y.push(y);
    ML.text.push(text);
    ML.align.push(r ? 'right' : 'left');
    ML.xo.push(r ? -10 : 10);
    ML.yo.push(up ? 12 : -12);
  };
  if (w.first >= 0)
    lab(G.xs[g][w.first], s * G.ys[g][w.first], `threshold reached, up ${s * G.ys[g][w.first]}`, true);
  if (w.danger >= 0) {
    const dx = G.xs[g][w.danger],
      dl = s * G.ys[g][w.danger];
    lab(dx, dl, `under ${SCARY}%: ${leadText(dl)} (${pc(hold(dl, dx, e))})`, false);
  }
  // Substitution markers retain their hover details without competing text labels.
  mark_labels.data = ML;
} else {
  turns.data = { x: [], y: [], on: [], marker: [], text: [] };
  mark_labels.data = { x: [], y: [], text: [], align: [], xo: [], yo: [] };
}
if (fg >= 0) {
  const [g, s] = pick.find((q) => q[0] === fg),
    [tx, ty] = G.st[g][s > 0 ? 'home' : 'away'],
    X = [],
    Y = [];
  for (let i = 0; i < tx.length; i++) {
    X.push(tx[i], i + 1 < tx.length ? tx[i + 1] : 0);
    Y.push(ty[i], ty[i]);
  }
  starters_src.data = { x: X, y: Y };
  strip_note.text = '';
} else {
  starters_src.data = { x: [], y: [] };
  strip_note.text = 'Pick a game to see when its starters sat';
}

game_axis.visible = fg < 0;
strip.visible = fg >= 0;

// ---- the game picker: the games drawn on the chart, in date order ----
// grouped by what happened, worst first; each game from the picked team's side: "Nov 01, vs TOR, W 131–125"
const GROUPS = [
  [BLOWN, 'Lost'],
  [SCARE, 'Close call, still won'],
  [HELD, 'Held'],
  [NEVER, 'Below threshold'],
];
const optLabel = (g, s) => {
  const me = s > 0 ? G.home[g] : G.away[g],
    opp = s > 0 ? G.away[g] : G.home[g];
  const mine = s > 0 ? G.hs[g] : G.as_[g],
    theirs = s > 0 ? G.as_[g] : G.hs[g];
  return `${G.date[g]}, ${all ? me + ' ' : ''}${s > 0 ? 'vs' : '@'} ${opp}, ${mine > theirs ? 'W' : 'L'} ${mine}–${theirs}${G.ot[g] ? ' (OT)' : ''}`;
};
const menu = new Map([
  [
    '',
    [
      [
        '-1',
        fg >= 0
          ? 'Show all games'
          : pick.length
            ? all
              ? 'Pick a game'
              : `Pick one of ${team}'s games`
            : 'No games match the filters',
      ],
    ],
  ],
]);
const order = [];
for (const [color, name] of GROUPS) {
  const qs = pick.filter((q) => q[3] === color).sort((a, b) => a[0] - b[0]);
  if (!qs.length) continue;
  menu.set(
    `${name} (${qs.length})`,
    qs.map(([g, s]) => [String(g), optLabel(g, s)]),
  );
  order.push(...qs.map((q) => q[0]));
}
game_pick.options = menu;
game_pick.tags = order;
clear_game.disabled = fg < 0;
if (game_pick.value !== String(fg)) game_pick.value = String(fg);
// ---- the view, in the address: a link opens the page exactly here ----
history.replaceState(
  null,
  '',
  '#' +
    new URLSearchParams({
      season: season.value,
      team,
      game: fg >= 0 ? G.id[fg] : '',
      lead: L,
      min: T,
      safe: pct,
      show: show.active.join(''),
      played: ['any', 'home', 'away'][where.active],
      matchup: matchup.active == null ? 'e' + Math.round(whatIf) : MATCHUP_KEYS[matchup.active],
    }).toString(),
);
""",
    )
    # a picked game that isn't among the new season's or team's games is dropped by update itself (Bokeh runs these
    # handlers after the code that set the value, so clearing it here would also wipe a game restored from a link)
    # a new team or all teams: the filter's opening state for it (registered first, so update then draws with it)
    # (tags: a filter restored from a link, which this team change must keep)
    team_select.js_on_change(
        "value",
        CustomJS(
            args=dict(show=outcome_filter),
            code=f"""
if (show.tags.length) {{ const keep = show.tags[0]; show.tags = []; show.active = keep; return; }}
const want = cb_obj.value === {ALL_TEAMS!r} ? [1, 2] : [0, 1, 2, 3];
if (show.active.join() !== want.join()) show.active = want;""",
        ),
    )
    for w in (season_select, team_select):
        w.js_on_change("value", update)
    for w in (lead_input, time_input):
        w.js_on_change("value", update)
    for w in (rest_threshold, matchup_control, outcome_filter, venue_filter):
        w.js_on_change("active", update)
    step_game = """
const order = pick.tags;
if (!order.length) return;
const i = order.indexOf(focus.data.g[0]); // nothing picked: start at the first (or last) game
focus.data = {
  g: [
    order[
      i < 0 ? (step > 0 ? 0 : order.length - 1) : Math.min(order.length - 1, Math.max(0, i + step))
    ],
  ],
};
update.execute(pick);
"""
    for b, step in ((prev_game, -1), (next_game, 1)):
        b.js_on_event(
            "button_click",
            CustomJS(
                args=dict(pick=game_pick, focus=selected_game, update=update, step=step),
                code=step_game,
            ),
        )
    game_pick.js_on_change(
        "value",
        CustomJS(
            args=dict(focus=selected_game, update=update, pick=game_pick),
            code="""
const g = Number(pick.value);
if (g === focus.data.g[0]) return; // set by update itself
focus.data = { g: [g] };
update.execute(pick);
""",
        ),
    )
    # the game chart: a click near the picked game's line asks panel 1 about that moment (its lead and time, with
    # that game's matchup); a click near another game's line picks that game
    game_lines.js_on_event(
        Tap,
        CustomJS(
            args=dict(
                on=game_line_source,
                focus=selected_game,
                update=update,
                lead=lead_input,
                left=time_input,
                matchup=matchup_control,
                ask_line=ask_line,
                firsts=rest_markers,
                dangers=close_call_markers,
            ),
            code="""
const x = cb_obj.x,
  y = cb_obj.y,
  d = on.data,
  fg = focus.data.g[0];
// a ● or ○ of another game: within about the mark's size (0.6 min across, 2 points up or down)
let mg = -1,
  md = 1;
for (const m of [firsts.data, dangers.data])
  for (let i = 0; i < m.x.length; i++) {
    if (!m.alpha[i] || m.g[i] === fg) continue;
    const k = Math.hypot((m.x[i] - x) / 0.6, (m.y[i] - y) / 2);
    if (k <= md) {
      mg = m.g[i];
      md = k;
    }
  }
if (mg >= 0) {
  focus.data = { g: [mg] };
  update.execute(on);
  return;
}
const at = (xs, ys, t) => {
  let v = ys[0];
  for (let i = 0; i < xs.length && xs[i] >= t; i++) v = ys[i];
  return v;
};
const near = (i) => Math.abs(at(d.xs[i], d.ys[i], x) - y);
const TOL = 2.5; // points: about the width of a line plus a little
const f = d.g.indexOf(fg);
if (f >= 0 && near(f) <= TOL) {
  const T = Math.min(24, Math.max(0.5, Math.round(x * 2) / 2)),
    L = at(d.xs[f], d.ys[f], T);
  if (x > 24.25 || L < 1) return; // the curves cover the second half, for a team that leads
  const Lc = Math.min(40, L);
  ask_line.tags = [`${fg}|${Lc}|${T}|${d.e[f]}`, T];
  matchup.tags = [d.e[f], d.label[f].split(':')[0]];
  if (matchup.active !== null) matchup.active = null; // these each rerun update; the last one draws the asked view
  lead.value = Lc;
  left.value = `${Math.floor(T)}:${T % 1 ? '30' : '00'}`;
  update.execute(lead);
  return;
}
let best = -1,
  bd = TOL;
for (let i = 0; i < d.g.length; i++) {
  const k = near(i);
  if (k <= bd) {
    best = i;
    bd = k;
  }
}
if (best >= 0 && d.g[best] !== fg) {
  focus.data = { g: [d.g[best]] };
  update.execute(on);
}
""",
        ),
    )
    # a circle on the curves: ask about that lead and time
    observed_source.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(src=observed_source, lead=lead_input, left=time_input),
            code="""
const i = src.selected.indices[0];
if (i === undefined) return;
src.selected.indices = [];
const T = src.data.x[i];
lead.value = src.data.lead[i];
left.value = `${Math.floor(T)}:${T % 1 ? '30' : '00'}`;
""",
        ),
    )
    # the waits histogram: hovering a bar lights up its games on the game chart; clicking picks the first of them
    light = CustomJS(
        args=dict(on=game_line_source, waits=waits_src),
        code="""
const i = cb_data && cb_data.index ? cb_data.index.indices[0] : undefined;
if (i === (waits.tags[0] ?? -1)) return;
waits.tags = [i ?? -1];
const d = on.data;
if (on.tags.length) Object.assign(d, { alpha: on.tags[0], width: on.tags[1] }); // back to update's own look
if (i === undefined) {
  on.tags = [];
  on.change.emit();
  return;
}
on.tags = [d.alpha, d.width];
const hit = new Set(waits.data.gs[i]);
on.data = {
  ...d,
  alpha: d.g.map((g, j) => (hit.has(g) ? 1 : 0.06)),
  width: d.g.map((g, j) => (hit.has(g) ? 2.5 : d.width[j])),
};
""",
    )
    waits.add_tools(
        HoverTool(renderers=[waits_hit], tooltips="@text", callback=light),
        TapTool(renderers=[waits_hit]),
    )
    waits.js_on_event(
        MouseLeave,
        CustomJS(
            args=dict(on=game_line_source, waits=waits_src),
            code="""
if (!on.tags.length) return;
waits.tags = [-1];
on.data = { ...on.data, alpha: on.tags[0], width: on.tags[1] };
on.tags = [];
""",
        ),
    )
    waits_src.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(src=waits_src, focus=selected_game, update=update),
            code="""
const i = src.selected.indices[0];
if (i === undefined) return;
src.selected.indices = [];
focus.data = { g: [src.data.gs[i][0]] };
update.execute(src);
""",
        ),
    )
    for src in (slope_source,):  # clicking a slope focuses that game; clicking it again clears
        src.selected.js_on_change(
            "indices",
            CustomJS(
                args=dict(src=src, focus=selected_game, update=update),
                code="""
const hits = src.selected.indices;
if (!hits.length) return;
const cur = focus.data.g[0],
  gs = hits.map((i) => src.data.g[i]);
src.selected.indices = [];
// a click that touches the focused game clears it (lines often overlap); otherwise focus the first game hit
focus.data = { g: [gs.includes(cur) ? -1 : gs[0]] };
update.execute(src);
""",
            ),
        )
    for p in (slope, team_bars_plot):
        p.select_one(TapTool).behavior = "select"
    team_bars_source.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(bars=team_bars_source, sel=team_select, ALL=ALL_TEAMS),
            code="""
const i = bars.selected.indices[0];
if (i === undefined) return;
const t = bars.data.team[i];
bars.selected.indices = [];
sel.value = sel.value === t ? ALL : t;
""",
        ),
    )

    # ---------- text ----------
    WIDTH = 1240
    section = lambda title: Div(width=WIDTH, **widget_style, text=f"<h2>{title}</h2>")
    header = Div(
        width=WIDTH - 120,
        **widget_style,
        text=f"""
<h1>How likely is an NBA lead to hold?</h1>
<div class="intro">Compare predicted outcomes with actual games and substitutions.</div>
<div class="space">
<div class="intro">A team's starting five, its <abbr title="{GLOSSARY['starters']}">starters</abbr>, play most of a
48-minute game. Resting them with a safe lead reduces fatigue and avoids injuries, but resting them too early can
allow the opponent to recover. We can estimate the probability that a lead holds from its size and the time
remaining, then compare it with observed outcomes and substitutions. Returning starters may help preserve those leads. The model reflects observed coaching, so reaching the threshold is not
a guarantee that starters can stay out.</div>
<div class="small muted">The summary covers both test seasons and follows the probability threshold, independently of the historical filters. Each historical game uses its own pre-game matchup.</div></div>
<div class="small muted">{" and ".join(TEST_SEASONS)} games through {games.game_date.max():%b %-d, %Y}</div>""",
    )
    decide = section("What does the model predict?")
    calculator_scope = Div(
        text='<div class="small muted">Adjust the lead, time, matchup, and probability threshold.</div>',
        **widget_style,
    )
    findings_scope = Div(
        text='<div class="small muted">Model benchmark and outcomes across both test seasons</div>',
        **widget_style,
    )
    history_scope = Div(
        text='<div class="small muted">Choose a season, team, or game to explore the historical results.</div>',
        **widget_style,
    )
    check = section("What happened in real games")

    footer = Div(
        width=WIDTH,
        **widget_style,
        text=f"""<div class="small muted space">
Model fit on {FIT_SEASONS} using score, time, and team strength (no possession, so the last 2 min aren't judged). Games shown: {" and ".join(TEST_SEASONS)},
never seen by the model. Statistics: <a href="https://www.nba.com/stats">NBA.com</a>. Preparation: <a href="https://github.com/azhangd/nba-analytics-duckdb">nba-analytics-duckdb</a>. Not affiliated with the NBA</div>""",
    )

    # Group scenario inputs above the probability estimate.
    lead_time_inputs = row(
        lead_input, time_input, styles={"gap": "16px"},
        stylesheets=[InlineStyleSheet(css=":host { flex-wrap: wrap !important; height: auto !important; }")],
    )
    scenario_inputs = column(
        lead_time_inputs,
        input_notice,
        column(matchup_title, matchup_control, matchup_note, styles={"gap": "0px"}),
        column(level_title, rest_threshold, styles={"gap": "0px"}),
        styles={"gap": "12px"},
        stylesheets=[InlineStyleSheet(css="""
@media (min-width: 968px) and (max-width: 1272px) {
  :host { display: grid !important; grid-template-columns: repeat(2, minmax(0, 1fr)); height: auto !important; }
  :host > .bk-Row { grid-column: 1 / -1; }
}
""")],
    )
    scenario_result = column(
        scenario_context,
        answer,
        styles={"gap": "4px", "border-top": f"1px solid {PANEL_EDGE}", "padding-top": "12px"},
    )
    controls = column(
        scenario_inputs,
        scenario_result,
        margin=(0, 0, 0, 5),
        styles={
            "gap": "16px", "padding": "12px 16px", "box-sizing": "border-box",
            "background": "transparent", "border": "1px solid #bbbbbd", "border-radius": "8px",
        },
    )
    for p in (
        probability_curves,
        game_lines,
    ):  # the toolbar shows only zoom, pan and reset; hover and click always stay on
        for t in p.toolbar.tools:
            t.visible = isinstance(t, (BoxZoomTool, PanTool, WheelZoomTool, ResetTool))
    # Stack rows below the width required by the desktop chart arrangement.
    stack_on_narrow = InlineStyleSheet(css="""
@media (max-width: 1272px) {
  :host { flex-direction: column !important; align-items: flex-start !important; height: auto !important; }
  :host > * { flex: none !important; }
}
""")
    wrap_controls = InlineStyleSheet(css=":host { flex-wrap: wrap !important; }")
    waits_panel = column(waits, waits_caption, styles={"gap": "2px"}, stylesheets=[help_height])
    lower_charts = row(
        slope, waits_panel, styles={"gap": f"{GAP}px"},
        stylesheets=[wrap_controls, help_height],
    )
    game_summary.stylesheets = [PAGE, help_height]
    game_charts = column(game_lines, strip, lower_charts, styles={"gap": "16px"}, stylesheets=[help_height])
    game_navigation = row(
        game_pick,
        clear_game,
        prev_game,
        next_game,
        styles={"gap": "4px", "align-items": "end"},
        stylesheets=[wrap_controls],
    )
    show_title.margin = where_title.margin = (0, 5, 0, 5)
    season_team_controls = row(season_select, team_select, width=530, styles={"gap": "16px"}, stylesheets=[wrap_controls, help_height])
    outcome_controls = column(show_title, outcome_filter, width=530, styles={"gap": "0px"})
    filters = column(
        row(
            season_team_controls,
            game_navigation,
            styles={"gap": "16px", "align-items": "end"},
            stylesheets=[wrap_controls],
        ),
        row(
            outcome_controls,
            column(where_title, venue_filter, styles={"gap": "0px"}),
            styles={"gap": "16px"}, stylesheets=[stack_on_narrow],
        ),
        styles={
            "gap": "8px", "padding-top": "8px", "padding-bottom": "12px",
        },
    )
    page = column(
        row(
            header,
            reset,
            styles={"gap": "16px", "align-items": "start"},
            stylesheets=[wrap_controls],
        ),
        decide,
        calculator_scope,
        row(
            controls,
            probability_curves,
            styles={"gap": "32px"},
            stylesheets=[stack_on_narrow],
        ),
        findings_scope,
        findings,
        level_note,
        check,
        history_scope,
        team_summary,
        filters,
        game_summary,
        row(
            game_charts,
            team_bars_plot,
            styles={"gap": f"{GAP}px"},
            stylesheets=[stack_on_narrow, help_height],
        ),
        footer,
        width=1240,
        styles={"margin": "0 auto"},
    )
    # every control back to how the page opens (the widgets' own callbacks redraw)
    reset.js_on_click(
        CustomJS(
            args=dict(
                update=update,
                season=season_select,
                sel=team_select,
                focus=selected_game,
                lead=lead_input,
                left=time_input,
                level=rest_threshold,
                matchup=matchup_control,
                ask_line=ask_line,
                show=outcome_filter,
                where=venue_filter,
                curves=probability_curves,
                lines=game_lines,
            ),
            code=f"""
curves.reset.emit(); lines.reset.emit();   // undo any zoom
season.value = {START_SEASON!r}; sel.value = {START_TEAM!r}; focus.data = {{g: [{initial_game_index}]}};
show.active = {outcome_filter.active}; where.active = {venue_filter.active};
lead.value = {lead_input.value}; left.value = {time_input.value!r}; level.active = {rest_threshold.active};
matchup.tags = []; matchup.active = {matchup_control.active}; ask_line.tags = ['', -99];
update.execute(sel);
""",
        )
    )
    doc = Document(theme=THEME)
    doc.add_root(page)
    doc.js_on_event(
        "document_ready",
        CustomJS(
            args=dict(
                page=page,
                header=header,
                findings=findings,
                decide=decide,
                check=check,
                calculator_scope=calculator_scope,
                findings_scope=findings_scope,
                history_scope=history_scope,
                filters=filters,
                team_summary=team_summary,
                game_summary=game_summary,
                footer=footer,
                curves=probability_curves,
                lines=game_lines,
                strip=strip,
                slope=slope,
                waits=waits,
                bars=team_bars_plot,
                game_charts=game_charts,
                lower_charts=lower_charts,
                waits_panel=waits_panel,
                waits_caption=waits_caption,
                controls=controls,
                scenario_inputs=scenario_inputs,
                lead_time_inputs=lead_time_inputs,
                scenario_result=scenario_result,
                navigation=game_navigation,
                season_team_controls=season_team_controls,
                outcome_controls=outcome_controls,
                outcome_filter=outcome_filter,
                game_pick=game_pick,
                scenario_widgets=[matchup_title, matchup_control, matchup_note, rest_threshold, scenario_context, answer, level_note, input_notice],
            ),
            code="""
// Canvas titles do not wrap automatically. Preserve the original wording on every resize.
const chartText = [curves, lines, strip, slope, waits, bars].flatMap(plot =>
  [...new Set([plot.title, ...plot.above.filter(item => item.type === 'Title')])]
    .map(title => ({plot, title, text: title.text}))
);
const chartHeights = new Map([curves, lines, strip, slope, waits, bars].map(plot => [plot, plot.height]));
const measure = document.createElement('canvas').getContext('2d');
const resize = () => {
  // 32 px leaves a 16 px margin on either side; 1240 px is the desktop layout.
  const width = Math.min(1240, window.innerWidth - 32);
  const stacked = window.innerWidth <= 1272;
  page.width = width;
  header.width = Math.max(280, width - 160);
  for (const block of [findings, decide, check, calculator_scope, findings_scope,
                       history_scope, filters, team_summary, footer]) block.width = width;
  controls.width = stacked && window.innerWidth >= 968 ? width - 5 : Math.min(464, width - 5);
  // Account for matching 16 px panel padding and the 1 px outline.
  lead_time_inputs.width = scenario_inputs.width = scenario_result.width = controls.width - 34;
  for (const widget of scenario_widgets) widget.width = controls.width < 464 ? controls.width - 44 : 430;
  season_team_controls.width = outcome_controls.width = Math.min(530, width);
  outcome_filter.width = Math.min(520, width - 10);
  game_pick.width = Math.min(320, width - 10);
  curves.width = stacked ? width : 739;
  const gameWidth = stacked ? width : 944;
  lines.width = strip.width = game_summary.width = game_charts.width = lower_charts.width = gameWidth;
  slope.width = 300;
  // Keep quarter landmarks on phones; omit closely packed end-of-game ticks.
  const gameTicks = gameWidth < 600 ? [48, 24, 0] : gameWidth < 944 ? [48, 36, 24, 12, 0] : [48, 36, 24, 12, 6, 2, 0];
  for (const plot of [lines, strip]) plot.below.find(item => item.type === 'LinearAxis').ticker.ticks = gameTicks;
  curves.below.find(item => item.type === 'LinearAxis').ticker.ticks = curves.width < 600 ? [24, 12, 0] : [24, 18, 12, 6, 2, 0];
  // Reflow all six legend entries instead of clipping the last ones.
  lines.above.find(item => item.type === 'Legend').ncols = gameWidth < 600 ? 1 : gameWidth < 944 ? 3 : 6;
  waits.width = gameWidth >= 944 ? gameWidth - 316 : gameWidth;
  waits_panel.width = waits_caption.width = waits.width;
  bars.width = stacked ? Math.min(width, 600) : 280;
  navigation.width = Math.min(width, 660);
  const extraTextHeight = new Map();
  for (const {plot, title, text} of chartText) {
    measure.font = `${title.text_font_style} ${title.text_font_size} ${title.text_font}`;
    // Reserve the 72 px axis gutter and 28 px toolbar/right margin.
    const available = plot.width - 100;
    const rows = [];
    let line = '';
    for (const word of text.split(/\\s+/)) {
      const next = line ? `${line} ${word}` : word;
      if (line && measure.measureText(next).width > available) {
        rows.push(line);
        line = word;
      } else line = next;
    }
    if (line) rows.push(line);
    title.text = rows.join('\\n');
    extraTextHeight.set(plot, (extraTextHeight.get(plot) || 0) + (rows.length - 1) * parseFloat(title.text_font_size) * title.text_line_height);
  }
  for (const [plot, height] of chartHeights) plot.height = Math.ceil(height + (extraTextHeight.get(plot) || 0));
  const legend = lines.above.find(item => item.type === 'Legend');
  lines.height += (Math.ceil(legend.items.length / legend.ncols) - 1) * (legend.label_height + legend.spacing);
};
// Wait for viewport layout before measuring text, and remeasure once fonts are ready.
let pendingResize = false;
const scheduleResize = () => {
  if (pendingResize) return;
  pendingResize = true;
  requestAnimationFrame(() => {
    pendingResize = false;
    resize();
  });
};
window.addEventListener('resize', scheduleResize);
let observedWidth;
const viewportObserver = new ResizeObserver(([entry]) => {
  if (entry.contentRect.width !== observedWidth) {
    observedWidth = entry.contentRect.width;
    scheduleResize();
  }
});
viewportObserver.observe(document.documentElement);
document.fonts.ready.then(scheduleResize);
resize();
""",
        ),
    )
    doc.js_on_event(
        "document_ready",
        CustomJS(
            args=dict(pick=game_pick, focus=selected_game, update=update, left=time_input),
            code="""
document.addEventListener('keydown', e => {
  if (e.altKey || e.ctrlKey || e.metaKey || !['ArrowLeft', 'ArrowRight'].includes(e.key)) return;
  const el = e.composedPath()[0];
  if (el && /^(INPUT|SELECT|TEXTAREA)$/.test(el.tagName)) return;   // typing in a box, or a menu open
  e.preventDefault();
  const step = e.key === 'ArrowRight' ? 1 : -1;"""
            + step_game.replace("return;", "return;", 1)
            + """
});
Bokeh.index.find_one(left).input_el.addEventListener('keydown', e => {
  const d = {ArrowUp: 0.5, ArrowDown: -0.5}[e.key];
  if (!d) return;
  e.preventDefault();
  const T = Math.min(24, Math.max(0.5, left.tags[0] + d));
  left.value = `${Math.floor(T)}:${T % 1 ? '30' : '00'}`;
});
""",
        ),
    )
    # on load: restore a view from the address (#season=...&team=...&game=...), then draw once (widget callbacks only
    # fire on change). The address is typed input: a value that doesn't name a real option is ignored.
    doc.js_on_event(
        "document_ready",
        CustomJS(
            args=dict(
                update=update,
                season=season_select,
                sel=team_select,
                focus=selected_game,
                lead=lead_input,
                left=time_input,
                level=rest_threshold,
                matchup=matchup_control,
                ask_line=ask_line,
                show=outcome_filter,
                where=venue_filter,
                LABELS=game_data["label"],
                IDS=game_data["id"],
                SEASON_OF=game_data["season"],
                RESTS=REST_THRESHOLDS,
                MATCHUP_KEYS=MATCHUP_KEYS,
            ),
            code="""
const h = new URLSearchParams(location.hash.slice(1));
const has = (sel, v) => sel.options.some((o) => (Array.isArray(o) ? o[0] : o) === v);
if (has(season, h.get('season'))) season.value = h.get('season');
const teamChanged = has(sel, h.get('team')) && h.get('team') !== sel.value;
if (has(sel, h.get('team'))) sel.value = h.get('team');
if (h.has('game')) {
  const g = IDS.indexOf(h.get('game'));
  focus.data = { g: [g >= 0 && SEASON_OF[g] === season.value ? g : -1] };
}
for (const [box, key] of [
  [lead, 'lead'],
  [left, 'min'],
]) {
  const v = parseFloat(h.get(key));
  if (Number.isFinite(v)) box.value = key === 'min' ? String(v) : v;
} // update cleans it
const r = RESTS.indexOf(parseFloat(h.get('safe')));
if (r >= 0) level.active = r;
if (/^[0-3]{0,4}$/.test(h.get('show') ?? 'x')) {
  show.active = [...new Set(h.get('show'))].map(Number).sort();
  if (teamChanged) show.tags = [show.active]; // the team's own handler runs after this one: it keeps these
}
const pl = ['any', 'home', 'away'].indexOf(h.get('played'));
if (pl >= 0) where.active = pl;
const mk = h.get('matchup') || '',
  m = MATCHUP_KEYS.indexOf(mk);
if (m >= 0) matchup.active = m;
else if (/^e-?\\d+$/.test(mk)) {
  // a moment asked about in a game: that game's own matchup, and its marker
  const g = focus.data.g[0],
    e = Number(mk.slice(1));
  matchup.tags = [e, g >= 0 ? LABELS[g].split(':')[0] : ''];
  matchup.active = null;
  const T = parseFloat(h.get('min')) || 4;
  if (g >= 0) ask_line.tags = [`${g}|${lead.value}|${T}|${e}`, T];
}
update.execute(sel);
""",
        ),
    )
    return file_html(doc, INLINE, title="How likely is an NBA lead to hold?")


if __name__ == "__main__":
    out = HERE / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(build())
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
