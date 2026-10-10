"""Build the project presentation (.pptx) from the project's own result files."""
import sys
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

project = Path(sys.argv[1])
output = Path(sys.argv[2])
reports = project / "reports"

validation = pd.read_csv(reports / "03_models" / "validation_comparison.csv")
test = pd.read_csv(reports / "03_models" / "test_results.csv").set_index("model")
backtest = pd.read_csv(reports / "04_backtest" / "backtest_summary.csv").set_index("metric")["value"]

MODEL = "Gradient boosting"
NAIVE = "No change from baseline"

# ------------------------------------------------------------------ style ---
INK = RGBColor(0x14, 0x17, 0x1A)
PAPER = RGBColor(0xFF, 0xFF, 0xFF)
RED = RGBColor(0xE1, 0x06, 0x00)
TEXT = RGBColor(0x1E, 0x22, 0x26)
MUTED = RGBColor(0x5B, 0x61, 0x68)
PANEL = RGBColor(0xF2, 0xF3, 0xF5)
GREY = RGBColor(0xB4, 0xB9, 0xC0)
SOFT_WHITE = RGBColor(0xC9, 0xCD, 0xD2)

HEAD = "Arial"
BODY = "Calibri"

deck = Presentation()
deck.slide_width = Inches(13.333)
deck.slide_height = Inches(7.5)
BLANK = deck.slide_layouts[6]

DECK_NAME = "F1 Race Strategy Simulator"


def text(slide, value, x, y, w, h, size=16, bold=False, color=TEXT, font=BODY,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False, spacing=None):
    """Add a text box. `value` is a string or a list of paragraphs."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = anchor

    paragraphs = value if isinstance(value, list) else [value]
    for index, content in enumerate(paragraphs):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = align
        if spacing is not None and index > 0:
            paragraph.space_before = Pt(spacing)
        run = paragraph.add_run()
        run.text = content
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.name = font
        run.font.color.rgb = color
    return box


def panel(slide, x, y, w, h, fill=PANEL):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.adjustments[0] = min(0.08, 0.14 / max(h, 0.1))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def number_dot(slide, number, x, y, size=0.46):
    """The deck's motif: a red circle carrying a step number."""
    dot = slide.shapes.add_shape(
        MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(size), Inches(size))
    dot.fill.solid()
    dot.fill.fore_color.rgb = RED
    dot.line.fill.background()
    dot.shadow.inherit = False
    frame = dot.text_frame
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.CENTER
    run = paragraph.add_run()
    run.text = str(number)
    run.font.size = Pt(15)
    run.font.bold = True
    run.font.name = HEAD
    run.font.color.rgb = PAPER


def background(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def content_slide(kicker, title, notes):
    """Light slide with the shared frame: kicker, title, footer, number."""
    slide = deck.slides.add_slide(BLANK)
    background(slide, PAPER)
    text(slide, kicker.upper(), 0.6, 0.42, 9, 0.3, size=12, bold=True, color=RED, font=HEAD)
    text(slide, title, 0.6, 0.74, 12.1, 0.75, size=32, bold=True, color=INK, font=HEAD,
         anchor=MSO_ANCHOR.MIDDLE)
    text(slide, DECK_NAME, 0.6, 6.98, 6, 0.26, size=10, color=MUTED)
    text(slide, str(len(deck.slides)), 11.73, 6.98, 1, 0.26, size=10, color=MUTED,
         align=PP_ALIGN.RIGHT)
    slide.notes_slide.notes_text_frame.text = notes
    return slide


def stat(slide, x, y, w, h, value, label, value_color=INK, value_size=44):
    panel(slide, x, y, w, h)
    text(slide, value, x + 0.28, y + 0.24, w - 0.56, 0.95, size=value_size, bold=True,
         color=value_color, font=HEAD, anchor=MSO_ANCHOR.MIDDLE)
    text(slide, label, x + 0.28, y + 1.28, w - 0.56, h - 1.45, size=14, color=MUTED)


def picture(slide, path, x, y, w):
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))


def style_chart(chart, number_format, highlight_index, n_points):
    chart.has_legend = False
    chart.has_title = False
    chart.font.size = Pt(12)
    chart.font.name = BODY
    chart.font.color.rgb = MUTED

    plot = chart.plots[0]
    plot.gap_width = 55
    plot.has_data_labels = True
    labels = plot.data_labels
    labels.number_format = number_format
    labels.number_format_is_linked = False
    labels.position = XL_LABEL_POSITION.OUTSIDE_END
    labels.font.size = Pt(12)
    labels.font.name = BODY
    labels.font.color.rgb = TEXT

    series = plot.series[0]
    for index in range(n_points):
        point = series.points[index]
        point.format.fill.solid()
        point.format.fill.fore_color.rgb = RED if index == highlight_index else GREY

    value_axis = chart.value_axis
    value_axis.visible = False
    value_axis.has_major_gridlines = False
    value_axis.minimum_scale = 0

    category_axis = chart.category_axis
    category_axis.has_major_gridlines = False
    category_axis.format.line.color.rgb = GREY
    category_axis.tick_labels.font.size = Pt(12)
    category_axis.tick_labels.font.color.rgb = TEXT


# ================================================================ slides ===

# 1 ---------------------------------------------------------------- title ---
slide = deck.slides.add_slide(BLANK)
background(slide, INK)
text(slide, "PREDICTIVE MODELLING · COURSE PROJECT", 0.7, 1.55, 6, 0.3, size=12,
     bold=True, color=RED, font=HEAD)
text(slide, "Can a model call the pit stop?", 0.7, 2.0, 5.6, 2.1, size=46, bold=True,
     color=PAPER, font=HEAD)
text(slide, "A race strategy simulator built on real Formula 1 lap data, 2022–2025",
     0.7, 4.35, 5.3, 1.0, size=18, color=SOFT_WHITE)
picture(slide, reports / "05_app" / "race_view.png", 6.65, 1.9, 6.1)
slide.notes_slide.notes_text_frame.text = (
    "This project asks whether a lap-time model, given only the first few laps of a "
    "race, can predict the rest of it well enough to choose a pit strategy. The image "
    "is the simulator we built on top of the model."
)

# 2 --------------------------------------------------------------- answer ---
slide = content_slide(
    "The answer first",
    "Race time is predictable; the exact pit lap is not",
    "Lead with the result. The model predicts total race time to about half a percent, "
    "less than half the error of assuming constant pace. But the recommended first pit "
    "lap is on average 8 laps from what the team did. The reason is on the last line: "
    "good strategies differ by a couple of seconds, and that is inside the model's error.",
)
mae_model = backtest["Race-time MAE on green races (s)"]
mae_naive = backtest["Race-time MAE with no model, constant baseline pace (s)"]
cards = [
    (f"{backtest['Race-time MAE on green races (% of race time)']:.2f}%",
     f"Race-time error for the strategy really used: {mae_model:.0f} s over a race of about 70 minutes", RED),
    (f"{mae_naive / mae_model:.1f}×",
     f"Lower race-time error than constant pace: {mae_model:.0f} s instead of {mae_naive:.0f} s", INK),
    (f"{backtest['Same number of stops as the team (%)']:.0f}%",
     "Recommended number of stops matches the team's", INK),
    (f"{backtest['First pit lap MAE vs team (laps)']:.0f} laps",
     "Average distance between the recommended first pit lap and the team's", INK),
]
for index, (value, label, color) in enumerate(cards):
    stat(slide, 0.6 + index * 3.07, 1.95, 2.87, 2.75, value, label, color)
panel(slide, 0.6, 5.1, 12.13, 1.3, fill=INK)
text(slide, "Good strategies differ by about "
     f"{backtest['Median predicted gain of recommended strategy (s)']:.0f} seconds. "
     "That is smaller than the model's own error, so it can rank races, not pit laps.",
     0.95, 5.1, 11.4, 1.3, size=19, color=PAPER, anchor=MSO_ANCHOR.MIDDLE)

# 3 ----------------------------------------------------------------- data ---
slide = content_slide(
    "Data",
    "Real race laps, split by season",
    "Data comes from the FastF1 library: lap times, tyre compound, tyre age, track "
    "status and weather. 92 races were downloaded; 12 wet races were dropped and 5 had "
    "no usable laps. The split is by season, not random, because laps in the same race "
    "are strongly correlated. A random split would leak the race into both sides.",
)
rows = [("Split", "Seasons", "Dry races", "Laps used"),
        ("Train", "2022, 2023", "35", "24,039"),
        ("Validation", "2024", "19", "15,332"),
        ("Test", "2025", "21", "15,844")]
table = slide.shapes.add_table(4, 4, Inches(0.6), Inches(1.95), Inches(6.6), Inches(2.6)).table
for r, row in enumerate(rows):
    for c, value in enumerate(row):
        cell = table.cell(r, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = INK if r == 0 else (PANEL if r % 2 else PAPER)
        cell.margin_left = Inches(0.15)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        paragraph = cell.text_frame.paragraphs[0]
        paragraph.alignment = PP_ALIGN.LEFT if c < 2 else PP_ALIGN.RIGHT
        run = paragraph.add_run()
        run.text = value
        run.font.size = Pt(16)
        run.font.name = BODY
        run.font.bold = r == 0 or c == 0
        run.font.color.rgb = PAPER if r == 0 else TEXT
text(slide, "Source: FastF1. 92 races downloaded; 12 wet races dropped; 5 had no "
     "usable laps after cleaning.", 0.6, 4.75, 6.6, 0.6, size=12, color=MUTED)
text(slide, "Only clean laps are modelled: green flag, dry tyres, no pit entry or "
     "exit, within 4% of the driver's median pace.", 0.6, 5.45, 6.6, 0.7, size=14,
     color=TEXT)

panel(slide, 7.6, 1.95, 5.13, 4.45)
text(slide, "Why split by season", 7.95, 2.2, 4.4, 0.4, size=20, bold=True, color=INK,
     font=HEAD)
text(slide, [
    "Laps from one race share the same track, weather and car. A random split puts "
    "the same race on both sides and overstates accuracy.",
    "Splitting by season means the model is always tested on races it has never seen.",
    "2022–2025 is one set of technical rules, so the cars are comparable.",
    "The 2025 season was used once, after the model was chosen.",
], 7.95, 2.8, 4.45, 3.5, size=16, color=TEXT, spacing=12)

# 4 --------------------------------------------------------------- target ---
slide = content_slide(
    "Target and inputs",
    "Predict the change in pace, not the lap time",
    "Absolute lap time is mostly the circuit and the car, which the model cannot know "
    "for a future race. So each lap is expressed relative to the driver's own baseline "
    "pace from their first clean laps. That baseline is known early in the race, so "
    "there is no leakage. Every input on the right is known in advance for any "
    "hypothetical strategy, which is what lets the model act as a simulator.",
)
panel(slide, 0.6, 1.95, 5.5, 4.45, fill=INK)
text(slide, "TARGET", 0.95, 2.2, 4.8, 0.3, size=12, bold=True, color=RED, font=HEAD)
text(slide, ["delta % =", "100 × (lap time ÷ baseline − 1)"], 0.95, 2.6, 4.8, 1.0,
     size=20, bold=True, color=PAPER, font=HEAD)
text(slide, [
    "Baseline = median of the driver's first clean laps (up to 8, between laps 2 and 12).",
    "It is known a few laps into the race, so predicting later laps from it uses no "
    "future information.",
    "Example: at a 90 s baseline, a lap predicted at +1.2% is 91.08 s.",
], 0.95, 3.7, 4.8, 2.55, size=15, color=SOFT_WHITE, spacing=10)

features = [
    ("Tyre age", "Laps completed on the current set"),
    ("Compound", "Soft, medium or hard"),
    ("Fuel burned", "Share of race distance since the baseline"),
    ("Race progress", "Share completed, and race length"),
    ("Baseline conditions", "Tyre and track temperature when pace was set"),
    ("Circuit", "Venue, as a category"),
]
for index, (name, detail) in enumerate(features):
    fx = 6.5 + (index % 2) * 3.17
    fy = 1.95 + (index // 2) * 1.52
    panel(slide, fx, fy, 2.97, 1.34)
    text(slide, name, fx + 0.22, fy + 0.18, 2.55, 0.35, size=16, bold=True, color=INK,
         font=HEAD)
    text(slide, detail, fx + 0.22, fy + 0.6, 2.55, 0.65, size=14, color=MUTED)

# 5 --------------------------------------------------------------- models ---
slide = content_slide(
    "Model comparison",
    "Small boosted trees narrowly beat linear regression",
    "Six candidates were compared on the 2024 validation season. The two naive "
    "references are at the bottom. Gradient boosting won, but only by about 3% over "
    "plain linear regression. Our first boosted model, with 31-leaf trees, actually "
    "lost to linear regression because it fitted race-specific noise. Shrinking the "
    "trees to 4 leaves fixed that.",
)
ordered = validation.sort_values("rmse_s", ascending=False).reset_index(drop=True)
chart_data = CategoryChartData()
chart_data.categories = list(ordered["model"])
chart_data.add_series("Validation RMSE", [round(v, 3) for v in ordered["rmse_s"]])
frame = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.6), Inches(2.2),
                               Inches(7.5), Inches(4.2), chart_data)
style_chart(frame.chart, '0.00" s"', list(ordered["model"]).index(MODEL), len(ordered))
text(slide, "Validation RMSE, seconds per lap (2024 season)", 0.6, 1.8, 7.5, 0.35,
     size=14, bold=True, color=INK)

best = validation.set_index("model")
panel(slide, 8.5, 1.95, 4.23, 4.45)
text(slide, "What the comparison shows", 8.82, 2.2, 3.6, 0.4, size=18, bold=True,
     color=INK, font=HEAD)
text(slide, [
    f"Gradient boosting: {best.loc[MODEL, 'rmse_s']:.2f} s per lap, selected.",
    f"Linear regression: {best.loc['Linear regression', 'rmse_s']:.2f} s. The gap is "
    "about 3%.",
    f"Assuming pace never changes: {best.loc[NAIVE, 'rmse_s']:.2f} s.",
    "Trees with 31 leaves fitted race-specific noise and lost to linear regression. "
    "Trees with 4 leaves won.",
], 8.82, 2.8, 3.6, 3.5, size=15, color=TEXT, spacing=11)

# 6 ----------------------------------------------------------------- test ---
slide = content_slide(
    "Test season",
    "On the unseen 2025 season the lap error is nearly halved",
    "The selected model was refitted on 2022 to 2024 and run once on 2025. Error drops "
    "from 1.56 to 0.83 seconds per lap. R squared is 0.44: traffic, driver management "
    "and track evolution are not in the data. The error splits into three levels. "
    "Only the stint-level part separates strategies; the driver-level part shifts all "
    "strategies for that driver equally. The chart is one driver's race predicted from "
    "their first laps only.",
)
stat(slide, 0.6, 1.95, 2.25, 2.1, f"{test.loc[MODEL, 'rmse_s']:.2f} s",
     "Lap error (RMSE) with the model", RED, value_size=32)
stat(slide, 3.05, 1.95, 2.25, 2.1, f"{test.loc[NAIVE, 'rmse_s']:.2f} s",
     "Lap error assuming constant pace", value_size=32)
panel(slide, 0.6, 4.2, 4.7, 2.3)
text(slide, f"Where the error sits (R² = {test.loc[MODEL, 'r2']:.2f})", 0.85, 4.35, 4.2,
     0.35, size=15, bold=True, color=INK, font=HEAD)
text(slide, [
    "Driver over a whole race: 0.55% of pace. Shifts every strategy equally.",
    "Stint within a race: 0.44%. This is what separates strategies.",
    "Single lap: 0.65%. Mostly averages out.",
], 0.85, 4.78, 4.25, 1.65, size=14, color=TEXT, spacing=4)
picture(slide, reports / "04_backtest" / "example_lap_trace.png", 5.7, 1.95, 7.03)
text(slide, "One driver's race, predicted from their first clean laps only.", 5.7, 5.9,
     7.03, 0.4, size=12, color=MUTED)

# 7 ------------------------------------------------------------ simulator ---
slide = content_slide(
    "Simulator",
    "Turning the model into a race simulator",
    "The model becomes a simulator in five steps. For any strategy we lay out which "
    "tyre is on the car and how old it is on every lap, predict each lap, add the "
    "circuit's pit-loss time per stop, and sum. We search every one-stop strategy and a "
    "grid of two-stop strategies, then draw random races from the model's measured "
    "error to show the spread. The search obeys three rules.",
)
steps = [
    ("Tyre plan", "Compound and tyre age for every lap of a strategy"),
    ("Predict laps", "Model output scaled by the driver's baseline pace"),
    ("Add pit loss", "Median green-flag loss at that circuit in past seasons"),
    ("Search", "Every one-stop plan and a grid of two-stop plans"),
    ("Uncertainty", "Random races drawn from stint and lap error"),
]
for index, (name, detail) in enumerate(steps):
    sx = 0.6 + index * 2.46
    panel(slide, sx, 1.95, 2.26, 2.45)
    number_dot(slide, index + 1, sx + 0.22, 2.17)
    text(slide, name, sx + 0.22, 2.8, 1.85, 0.35, size=16, bold=True, color=INK, font=HEAD)
    text(slide, detail, sx + 0.22, 3.22, 1.85, 1.1, size=14, color=MUTED)

text(slide, "Rules the search enforces", 0.6, 4.75, 6, 0.35, size=16, bold=True,
     color=INK, font=HEAD)
rules = [
    "At least two dry compounds in the race",
    "Every stint at least 5 laps long",
    "No stint longer than tyres were seen to last at that circuit",
]
for index, rule in enumerate(rules):
    rx = 0.6 + index * 4.1
    panel(slide, rx, 5.25, 3.93, 1.15, fill=INK)
    text(slide, rule, rx + 0.25, 5.25, 3.45, 1.15, size=14, color=PAPER,
         anchor=MSO_ANCHOR.MIDDLE)

# 8 ------------------------------------------------------------- backtest ---
slide = content_slide(
    "Backtest",
    "Checked against 182 real driver-races from 2025",
    "We ran the simulator for every top-ten finisher in the dry 2025 races. Race time "
    "could be checked on the 60 driver-races that stayed fully green, because safety "
    "cars are not modelled. There the error is 24 seconds against 55 with no model. "
    "The scatter compares the recommended first pit lap with the team's: points far "
    "from the diagonal are disagreements, and most recommendations are later than the "
    "team's stop.",
)
chart_data = CategoryChartData()
chart_data.categories = ["Constant pace", "Model"]
chart_data.add_series("Race-time MAE", [round(mae_naive, 1), round(mae_model, 1)])
frame = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(2.35),
                               Inches(3.6), Inches(3.5), chart_data)
style_chart(frame.chart, '0.0" s"', 1, 2)
text(slide, "Race-time error (MAE), seconds", 0.6, 1.9, 3.6, 0.35, size=14, bold=True,
     color=INK)
text(slide, "60 driver-races that stayed under green flags.", 0.6, 5.95, 3.6, 0.45,
     size=12, color=MUTED)

picture(slide, reports / "04_backtest" / "first_pit_lap_recommended_vs_actual.png",
        4.6, 1.9, 4.45)

side = [
    (f"{backtest['Same number of stops as the team (%)']:.0f}%", "same number of stops as the team"),
    (f"{backtest['First pit lap MAE vs team (laps)']:.1f} laps", "average distance of the first pit lap"),
    (f"{backtest['First pit lap within 3 laps of team (%)']:.0f}%", "of first pit laps within 3 laps"),
]
for index, (value, label) in enumerate(side):
    sy = 1.9 + index * 1.52
    panel(slide, 9.45, sy, 3.28, 1.36)
    text(slide, value, 9.7, sy + 0.14, 2.8, 0.6, size=28, bold=True, color=INK, font=HEAD,
         anchor=MSO_ANCHOR.MIDDLE)
    text(slide, label, 9.7, sy + 0.8, 2.8, 0.45, size=14, color=MUTED)

# 9 ----------------------------------------------------------------- demo ---
slide = content_slide(
    "Demo",
    "The simulator as an animated race",
    "This is the app. The same driver runs three strategies at once on the real "
    "circuit outline: the recommended one, the team's real one, and one the user sets. "
    "Wheel colour is the tyre compound, cars drive into the pit box during a stop, and "
    "the panel shows live gaps. A random mode draws one possible race from the model's "
    "error, so the finishing order can change between runs. Switch to the live app here.",
)
picture(slide, reports / "05_app" / "race_view.png", 0.6, 1.9, 8.0)
points = [
    ("Three strategies race", "Recommended, the team's real one, and your own"),
    ("Tyres and stops are visible", "Wheel colour is the compound; cars use the pit box"),
    ("Live running order", "Gaps, tyre age and a lead chart, lap by lap"),
    ("Random races", "Each run draws from the model's measured error"),
]
for index, (name, detail) in enumerate(points):
    py = 1.9 + index * 1.13
    number_dot(slide, index + 1, 8.95, py + 0.02, size=0.42)
    text(slide, name, 9.55, py, 3.2, 0.32, size=15, bold=True, color=INK, font=HEAD)
    text(slide, detail, 9.55, py + 0.36, 3.2, 0.62, size=14, color=MUTED)
text(slide, "Circuit outlines are real. Start-line position and direction of travel "
     "are illustrative.", 0.6, 6.45, 8.0, 0.35, size=12, color=MUTED)

# 10 ------------------------------------------------------------- limits ---
slide = content_slide(
    "Limitations",
    "Why the pit lap is hard, and what would fix it",
    "Four limits explain the gap to the teams. Long stints only appear in the data "
    "when the tyres held up, so the model is optimistic about old tyres and recommends "
    "later stops. Teams stop early for track position, which we do not model. Safety "
    "cars decide many races. And the gains between good strategies are inside the "
    "model's error. The next steps follow directly from these.",
)
limits = [
    ("Survivorship in tyre data",
     "Long stints are only seen when the tyres lasted, so old tyres look better than "
     "they are and stops come out too late."),
    ("No traffic or track position",
     "Teams stop early to undercut a rival or rejoin in clear air. The simulator only "
     "finds the fastest plan in clean air."),
    ("No safety cars or weather",
     "These decide many real races. Race time could only be checked on races that "
     "stayed green."),
    ("Gains are inside the error",
     "Good strategies differ by about 2 s; stint-level error alone is larger. The "
     "team's choice is a reference, not proven best."),
]
for index, (name, detail) in enumerate(limits):
    lx = 0.6 + (index % 2) * 6.16
    ly = 1.95 + (index // 2) * 1.78
    panel(slide, lx, ly, 5.97, 1.6)
    number_dot(slide, index + 1, lx + 0.25, ly + 0.25)
    text(slide, name, lx + 0.95, ly + 0.22, 4.8, 0.35, size=16, bold=True, color=INK,
         font=HEAD)
    text(slide, detail, lx + 0.95, ly + 0.62, 4.8, 0.9, size=14, color=MUTED)
panel(slide, 0.6, 5.6, 12.13, 0.9, fill=INK)
text(slide, "Next: add safety-car scenarios, separate the three compounds better, "
     "and model traffic.", 0.95, 5.6, 11.4, 0.9, size=16, color=PAPER,
     anchor=MSO_ANCHOR.MIDDLE)

# 11 ------------------------------------------------------------ closing ---
slide = deck.slides.add_slide(BLANK)
background(slide, INK)
text(slide, "CONCLUSION", 0.9, 1.5, 6, 0.3, size=12, bold=True, color=RED, font=HEAD)
text(slide, "An accurate model is not yet a decision", 0.9, 1.95, 11.5, 1.2, size=42,
     bold=True, color=PAPER, font=HEAD)
text(slide, [
    f"Lap time: {test.loc[MODEL, 'rmse_s']:.2f} s error on an unseen season, about "
    "half the error of constant pace.",
    f"Race time: within {backtest['Race-time MAE on green races (% of race time)']:.2f}% "
    "for the strategy really used.",
    "Pit lap: the choice the simulator exists for turns on differences of about "
    "2 seconds, which the model cannot resolve.",
], 0.9, 3.45, 9.6, 2.2, size=19, color=SOFT_WHITE, spacing=12)
text(slide, "Validate a model at the level of the decision it will be used for.",
     0.9, 5.85, 11, 0.5, size=20, bold=True, color=PAPER, font=HEAD)
slide.notes_slide.notes_text_frame.text = (
    "The lesson for predictive modelling: pointwise accuracy and even system-level "
    "accuracy were both good, but the decision needs a finer resolution than the model "
    "has. Accuracy has to be judged at the level of the decision. Questions."
)

output.parent.mkdir(parents=True, exist_ok=True)
deck.save(output)
print("saved", output, "slides:", len(deck.slides))
