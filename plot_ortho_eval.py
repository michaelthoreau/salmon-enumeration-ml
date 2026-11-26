"""Builds the README figures for the ortho evaluation from eval_ortho_local.py outputs.

Run after:
  python eval_ortho_local.py ... --zooms 21,22 --gsd_mm 9.6,7.4 --tta d4 --ensembles 7.4+9.6 --out debug/ortho_eval_tta
  python eval_ortho_local.py ... --score_only --overlay_methods stitched_9.6mm_rot4 --overlay_thresh 0.2 --out debug/ortho_eval_tta
  python eval_ortho_local.py ... --score_only --overlay_methods endpoint_z21 --overlay_thresh 0.25 --out debug/ortho_eval_tta
"""
import csv
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC, OUT = "debug/ortho_eval_tta", "debug/ortho_eval_figures"
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)

# --- side-by-side overlay comparison --------------------------------------
best = {}
for r in csv.DictReader(open(f"{SRC}/summary.csv")):
    best.setdefault(r["method"], {})[float(r["score_thresh"])] = r
def stats(m, th):
    r = best[m][th]
    return f"{r['tp']}/{r['gt']} fish found · {r['fp']} false positives (thr {th})"
panels = [("endpoint_z21", "Current /predict_ortho (falls back to zoom 21)", 0.25),
          ("stitched_9.6mm_rot4", "Stitched 9.6 mm + 4-rotation TTA", 0.2)]
for k in (1,):  # the crop used in the README
    imgs = []
    for m, title, th in panels:
        im = Image.open(f"{SRC}/overlay_{k}_{m}.jpg").convert("RGB")
        im = im.crop((0, 90, im.width, im.height)).resize((1100, round(1100 * (im.height - 90) / im.width)))
        head = Image.new("RGB", (im.width, 110), (20, 20, 20))
        d = ImageDraw.Draw(head)
        d.text((18, 14), title, font=font, fill=(255, 255, 255))
        d.text((18, 62), f"Labelled area: {stats(m, th)}", font=small, fill=(200, 200, 200))
        panel = Image.new("RGB", (im.width, im.height + 110))
        panel.paste(head, (0, 0)); panel.paste(im, (0, 110))
        imgs.append(panel)
    gap = 12
    legend_h = 60
    out = Image.new("RGB", (sum(i.width for i in imgs) + gap, imgs[0].height + legend_h), (20, 20, 20))
    x = 0
    for i in imgs:
        out.paste(i, (x, 0)); x += i.width + gap
    d = ImageDraw.Draw(out)
    y = imgs[0].height + 14
    x = 18
    for colour, label, shape in [((0, 255, 0), "detected fish (true positive)", "o"),
                                 ((255, 215, 0), "labelled fish the model missed", "o"),
                                 ((255, 40, 40), "false positive", "s")]:
        box = [x, y + 4, x + 26, y + 30]
        (d.ellipse if shape == "o" else d.rectangle)(box, outline=colour, width=4)
        d.text((x + 38, y), label, font=small, fill=(230, 230, 230))
        x += 60 + d.textlength(label, font=small)
    out.save(f"{OUT}/comparison_{k}.jpg", quality=85)

# --- F1 / AP bar chart -----------------------------------------------------
rows = [("Production: endpoint, zoom 21 fallback", "endpoint_z21", "endpoint"),
        ("Endpoint at zoom 22", "endpoint_z22", "endpoint"),
        ("Stitched 9.6 mm", "stitched_9.6mm_none", "stitched"),
        ("Stitched 9.6 mm + flip TTA", "stitched_9.6mm_flip", "stitched"),
        ("Stitched 9.6 mm + 4-rotation TTA", "stitched_9.6mm_rot4", "stitched"),
        ("Stitched 7.4 mm + 8-way TTA", "stitched_7.4mm_d4", "stitched"),
        ("Ensemble 7.4 + 9.6 mm, 8-way TTA", "ens_7.4+9.6", "stitched")]
f1 = [max(float(r["f1"]) for r in best[m].values()) for _, m, _ in rows]
colours = {"endpoint": "#eb6834", "stitched": "#2a78d6"}
fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
y = list(range(len(rows)))[::-1]
ax.barh(y, f1, height=0.62, color=[colours[g] for *_, g in rows], edgecolor="#fcfcfb", linewidth=2)
for yi, v in zip(y, f1):
    ax.text(v + 0.012, yi, f"{v:.2f}", va="center", fontsize=10, color="#0b0b0b")
ax.set_yticks(y, [r[0] for r in rows], fontsize=10, color="#0b0b0b")
ax.set_xlim(0, 1); ax.set_xlabel("Best F1 (match radius 0.5 m)", color="#52514e")
ax.tick_params(colors="#52514e", length=0)
ax.grid(axis="x", color="#e4e3df", linewidth=0.8); ax.set_axisbelow(True)
for s in ax.spines.values(): s.set_visible(False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=colours["endpoint"], label="Current per-tile endpoint"),
                   Patch(color=colours["stitched"], label="Stitched, overlapping windows")],
          loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9)
fig.suptitle("Ortho fish detection, 164 labelled fish", x=0.02, ha="left", fontsize=12, color="#0b0b0b")
fig.tight_layout(); fig.savefig(f"{OUT}/f1_by_method.png", facecolor=fig.get_facecolor())
