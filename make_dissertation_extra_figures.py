from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
TABLE_DIR = ROOT / "outputs" / "dissertation_analysis_tables"
FIG_DIR = ROOT / "outputs" / "dissertation_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT = font(28)
FONT_SMALL = font(22)
FONT_TINY = font(18)
FONT_BOLD = font(34, bold=True)


COLORS = {
    "ink": (35, 45, 55),
    "muted": (120, 135, 155),
    "grid": (218, 224, 232),
    "green": (42, 157, 143),
    "red": (192, 84, 84),
    "blue": (69, 123, 157),
    "navy": (31, 42, 55),
    "orange": (232, 154, 74),
    "purple": (126, 92, 172),
    "teal_light": (190, 229, 223),
    "red_light": (238, 202, 202),
}


def canvas(width=1800, height=950):
    return Image.new("RGB", (width, height), "white")


def draw_axes(draw, box, y_min, y_max, x_labels, y_label=None, label_every=1):
    x0, y0, x1, y1 = box
    draw.line((x0, y1, x1, y1), fill=COLORS["ink"], width=3)
    draw.line((x0, y0, x0, y1), fill=COLORS["ink"], width=3)
    for i in range(5):
        v = y_min + (y_max - y_min) * i / 4
        y = y1 - (v - y_min) / (y_max - y_min) * (y1 - y0)
        draw.line((x0, y, x1, y), fill=COLORS["grid"], width=2)
        draw.text((x0 - 85, y - 14), f"{v:.2f}", fill=COLORS["ink"], font=FONT_TINY)
    n = len(x_labels)
    xs = [x0 + i * (x1 - x0) / (n - 1) for i in range(n)]
    for idx, (x, label) in enumerate(zip(xs, x_labels)):
        draw.line((x, y1, x, y1 + 8), fill=COLORS["ink"], width=2)
        if idx % label_every == 0:
            draw.text((x - 40, y1 + 20), label, fill=COLORS["ink"], font=FONT_TINY)
    if y_label:
        draw.text((x0 - 105, y0 - 45), y_label, fill=COLORS["ink"], font=FONT_SMALL)
    return xs


def y_pos(v, y_min, y_max, box):
    return box[3] - (v - y_min) / (y_max - y_min) * (box[3] - box[1])


def draw_line(draw, xs, values, y_min, y_max, box, color, width=6, marker=9):
    pts = [(x, y_pos(v, y_min, y_max, box)) for x, v in zip(xs, values)]
    if len(pts) > 1:
        draw.line(pts, fill=color, width=width, joint="curve")
    for x, y in pts:
        draw.ellipse((x - marker, y - marker, x + marker, y + marker), fill=color)


def draw_legend(draw, items, x, y):
    offset = 0
    for label, color in items:
        draw.line((x + offset, y + 10, x + offset + 45, y + 10), fill=color, width=8)
        draw.ellipse((x + offset + 17, y + 1, x + offset + 29, y + 13), fill=color)
        draw.text((x + offset + 58, y - 3), label, font=FONT_SMALL, fill=COLORS["ink"])
        offset += 310


def save(img, name):
    png = FIG_DIR / f"{name}.png"
    img.save(png, dpi=(300, 300))
    return png


def five_year_robustness():
    df = pd.read_csv(TABLE_DIR / "five_year_robustness_summary.csv")
    df = df[df["five_year_period"].str[:4].astype(int) >= 1840].copy()
    img = canvas(1900, 980)
    draw = ImageDraw.Draw(img)
    draw.text((95, 60), "Figure 7. Five-year robustness of net sentiment", font=FONT_BOLD, fill=COLORS["ink"])
    draw.text((95, 112), "The broad U-shaped pattern is visible when the corpus is aggregated into five-year blocks rather than decades.", font=FONT_SMALL, fill=COLORS["muted"])
    box = (150, 210, 1760, 760)
    vals = df["sentiment_pos_minus_neg_mean"].tolist()
    labels = [f"{y}s" for y in df["five_year_period"].str[:4].tolist()]
    y_min, y_max = min(vals) - 0.04, max(vals) + 0.04
    xs = draw_axes(draw, box, y_min, y_max, labels, "Positive minus negative", label_every=2)
    draw_line(draw, xs, vals, y_min, y_max, box, COLORS["navy"], width=6, marker=8)
    draw.text((150, 825), "Note: labels are shown every decade for readability; points are five-year blocks.", font=FONT_TINY, fill=COLORS["muted"])
    return save(img, "fig6_five_year_net_sentiment_robustness")


def adjusted_trends():
    df = pd.read_csv(TABLE_DIR / "adjusted_decade_trends.csv")
    sent = df[df["outcome"] == "sentiment_pos_minus_neg"].copy()
    dist = df[df["outcome"] == "emotion_distress"].copy()
    img = canvas(2000, 980)
    draw = ImageDraw.Draw(img)
    draw.text((95, 55), "Figure 8. Raw and adjusted decade trends", font=FONT_BOLD, fill=COLORS["ink"])
    draw.text((95, 107), "Adjusted values control descriptively for article length, readability and publisher composition.", font=FONT_SMALL, fill=COLORS["muted"])

    for idx, (title, dat, ymin, ymax) in enumerate([
        ("a  Net sentiment", sent, -0.06, 0.20),
        ("b  Distress language", dist, 0.25, 0.62),
    ]):
        left = 140 + idx * 930
        box = (left, 225, left + 780, 755)
        labels = [f"{int(d)}s" for d in dat["decade"]]
        xs = draw_axes(draw, box, ymin, ymax, labels)
        draw.text((left, 165), title, font=FONT_BOLD, fill=COLORS["ink"])
        draw_line(draw, xs, dat["raw_mean"].tolist(), ymin, ymax, box, COLORS["muted"], width=5, marker=8)
        draw_line(draw, xs, dat["adjusted_mean"].tolist(), ymin, ymax, box, COLORS["green"], width=6, marker=9)
    draw_legend(draw, [("Raw mean", COLORS["muted"]), ("Adjusted mean", COLORS["green"])], 680, 845)
    return save(img, "fig7_raw_adjusted_decade_trends")


def publisher_composition():
    pub = pd.read_csv(TABLE_DIR / "publisher_composition_by_decade_top10.csv")
    totals = pd.read_csv(TABLE_DIR / "decade_descriptive_statistics.csv")[["decade", "n_articles"]]
    def simple_publisher(value):
        value = str(value).split(". [volume]")[0]
        value = value.split(" [volume]")[0]
        value = value.replace("[volume]", "").strip()
        value = value.rstrip(".")
        mapping = {
            "Daily evening star": "Evening star",
            "Evening star": "Evening star",
            "The New York herald": "The New York herald",
            "New-York tribune": "New-York tribune",
            "The Washington times": "The Washington times",
            "The St. Louis Republic": "The St. Louis republic",
        }
        return mapping.get(value, value)

    pub["publisher_group"] = pub["publisher"].map(simple_publisher)
    grouped = (
        pub.groupby(["decade", "publisher_group"], as_index=False)["articles"].sum()
    )
    top_publishers = (
        grouped.groupby("publisher_group")["articles"].sum().sort_values(ascending=False).head(5).index.tolist()
    )
    decades = sorted(totals["decade"].tolist())
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"], COLORS["red"]]
    img = canvas(1900, 980)
    draw = ImageDraw.Draw(img)
    draw.text((95, 55), "Figure 9. Publisher composition of the retained corpus", font=FONT_BOLD, fill=COLORS["ink"])
    draw.text((95, 107), "Stacked bars show the five largest retained publishers overall; all remaining publishers are grouped as Other.", font=FONT_SMALL, fill=COLORS["muted"])
    box = (150, 220, 1700, 780)
    x0, y0, x1, y1 = box
    draw.line((x0, y1, x1, y1), fill=COLORS["ink"], width=3)
    draw.line((x0, y0, x0, y1), fill=COLORS["ink"], width=3)
    for i in range(0, 101, 25):
        y = y1 - i / 100 * (y1 - y0)
        draw.line((x0, y, x1, y), fill=COLORS["grid"], width=2)
        draw.text((x0 - 70, y - 13), f"{i}%", font=FONT_TINY, fill=COLORS["ink"])
    width = (x1 - x0) / len(decades) * 0.68
    for j, dec in enumerate(decades):
        cx = x0 + (j + 0.5) * (x1 - x0) / len(decades)
        total = totals.loc[totals.decade == dec, "n_articles"].iloc[0]
        bottom = y1
        used = 0
        for pub_name, color in zip(top_publishers, colors):
            articles = grouped[(grouped.decade == dec) & (grouped.publisher_group == pub_name)]["articles"].sum()
            share = articles / total * 100
            used += share
            top = bottom - share / 100 * (y1 - y0)
            draw.rectangle((cx - width / 2, top, cx + width / 2, bottom), fill=color)
            bottom = top
        other = max(0, 100 - used)
        top = bottom - other / 100 * (y1 - y0)
        draw.rectangle((cx - width / 2, top, cx + width / 2, bottom), fill=(205, 213, 224))
        draw.text((cx - 32, y1 + 20), f"{dec}s", font=FONT_TINY, fill=COLORS["ink"])
    legend_y = 830
    legend_items = [(p.split(". [volume]")[0][:40], c) for p, c in zip(top_publishers, colors)] + [("Other", (205, 213, 224))]
    x = 150
    for label, color in legend_items:
        draw.rectangle((x, legend_y, x + 28, legend_y + 22), fill=color)
        draw.text((x + 38, legend_y - 4), label, font=FONT_TINY, fill=COLORS["ink"])
        x += min(300, 140 + len(label) * 8)
        if x > 1600:
            x = 150
            legend_y += 38
    return save(img, "fig8_publisher_composition")


def long_text_audit():
    df = pd.read_csv(TABLE_DIR / "long_text_truncation_audit.csv")
    img = canvas(1800, 930)
    draw = ImageDraw.Draw(img)
    draw.text((95, 55), "Figure 10. Long-text exposure under transformer input limits", font=FONT_BOLD, fill=COLORS["ink"])
    draw.text((95, 107), "Only a minority of retained articles exceed 512 or 1,000 words, but the share is not constant across decades.", font=FONT_SMALL, fill=COLORS["muted"])
    box = (150, 220, 1640, 730)
    labels = [f"{int(d)}s" for d in df["decade"]]
    vals512 = df["share_word_count_gt_512_percent"].tolist()
    vals1000 = df["share_word_count_gt_1000_percent"].tolist()
    y_max = max(vals512) + 2
    xs = draw_axes(draw, box, 0, y_max, labels, "Share of articles (%)")
    draw_line(draw, xs, vals512, 0, y_max, box, COLORS["blue"], width=6, marker=8)
    draw_line(draw, xs, vals1000, 0, y_max, box, COLORS["red"], width=6, marker=8)
    draw_legend(draw, [(">512 words", COLORS["blue"]), (">1,000 words", COLORS["red"])], 600, 800)
    return save(img, "fig9_long_text_truncation_audit")


def workbook():
    out = TABLE_DIR / "dissertation_analysis_tables.xlsx"
    files = [
        "corpus_stage_comparison_totals.csv",
        "corpus_stage_comparison_by_decade_percent.csv",
        "decade_descriptive_statistics.csv",
        "decade_cluster_bootstrap_confidence_intervals.csv",
        "five_year_robustness_summary.csv",
        "adjusted_decade_trends.csv",
        "publisher_composition_by_decade_top10.csv",
        "content_type_summary_by_decade.csv",
        "long_text_truncation_audit.csv",
        "manual_validation_sample_by_decade.csv",
    ]
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        for f in files:
            sheet = Path(f).stem[:31]
            pd.read_csv(TABLE_DIR / f).to_excel(writer, sheet_name=sheet, index=False)
    return out


if __name__ == "__main__":
    outputs = [
        five_year_robustness(),
        adjusted_trends(),
        publisher_composition(),
        long_text_audit(),
        workbook(),
    ]
    for path in outputs:
        print(path)
