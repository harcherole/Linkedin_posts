"""
MACHINE LEARNING ALGORITHMS — Animation fond blanc premium
10 scènes : algorithmes ML classiques avec visualisation intuitive + code Python
Fond blanc, palette professionnelle, style éditorial
Suivez-moi sur YouTube : https://www.youtube.com/@DIAM-IA
"""

import os, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm, multivariate_normal
from scipy.special import softmax
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
OUT          = "gifs/ml_algorithms.gif"
STAGE_FRAMES = 52
PAUSE_FRAMES = 12
TOTAL        = 10
FPS          = 20
DPI          = 140
os.makedirs("gifs", exist_ok=True)

FOOTER = ("LinkedIn : Georf MIGUIAMA BAMBA   |   "
          "TikTok : georf_tech   |   YouTube : D.I.A.M IA")

# ── Palette FOND BLANC ─────────────────────────────────────────────────────────
BG    = "#FFFFFF"
PANEL = "#F8F9FC"
GRID  = "#EEF0F6"
BORD  = "#CBD5E1"
INK   = "#0F172A"
INK2  = "#334155"
MUTED = "#94A3B8"

# Accents professionnels
BLUE   = "#2563EB"
INDIGO = "#4F46E5"
VIOLET = "#7C3AED"
GREEN  = "#059669"
TEAL   = "#0891B2"
ORANGE = "#EA580C"
RED    = "#DC2626"
AMBER  = "#D97706"
PINK   = "#DB2777"
LIME   = "#65A30D"

# Code style
CODE_BG  = "#0F172A"
CODE_FG  = "#E2E8F0"
CODE_STR = "#86EFAC"
CODE_KW  = "#93C5FD"
CODE_FN  = "#FCA5A5"
CODE_CMT = "#64748B"
CODE_NUM = "#FCD34D"

# Couleurs par scène
SCENE_COLS = [BLUE, GREEN, ORANGE, VIOLET, TEAL,
              RED, INDIGO, AMBER, PINK, LIME]

# ── Helpers ────────────────────────────────────────────────────────────────────
def ease(t):    return t*t*(3-2*t)
def cl(v):      return max(0.0, min(1.0, v))
def ph(ts, t0, t1):
    if t1<=t0: return 1.0 if ts>=t0 else 0.0
    return ease(cl((ts-t0)/(t1-t0)))

def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORD); sp.set_linewidth(0.8)
    ax.grid(True, color=GRID, linewidth=0.55, zorder=0)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)

def title_band(ax, num, title, subtitle, col):
    ax.text(0.01, 0.998, f"{num}/10", transform=ax.transAxes,
            fontsize=7.5, color=col, va="top", fontweight="bold",
            fontfamily="monospace", zorder=20)
    ax.text(0.50, 0.998, title, transform=ax.transAxes,
            fontsize=11, color=INK, va="top", ha="center",
            fontfamily="monospace", fontweight="bold", zorder=20)
    ax.text(0.50, 0.966, subtitle, transform=ax.transAxes,
            fontsize=7.8, color=col, va="top", ha="center",
            fontfamily="monospace", zorder=20)

def code_snippet(ax, x, y, lines, w=4.6, lh=0.29):
    """Bloc code style VSCode sombre."""
    h = len(lines)*lh + 0.22
    rect = FancyBboxPatch((x, y-h), w, h,
                          boxstyle="round,pad=0.05",
                          facecolor=CODE_BG, edgecolor="#1E293B",
                          linewidth=1.0, zorder=8)
    ax.add_patch(rect)
    for di, dc in enumerate(["#EF4444","#F59E0B","#22C55E"]):
        ax.add_patch(plt.Circle((x+0.11+di*0.16, y-0.13),
                                0.048, color=dc, zorder=9))
    for li, (txt, col) in enumerate(lines):
        ax.text(x+0.11, y-0.26-li*lh, txt,
                fontsize=7.2, color=col, fontfamily="monospace",
                va="top", zorder=9)

def glow_pts(ax, x, y, col, s=30, a=0.6):
    ax.scatter(x, y, s=s*3, color=col, alpha=0.12, zorder=3, edgecolors="none")
    ax.scatter(x, y, s=s,   color=col, alpha=a,    zorder=4, edgecolors="white", lw=0.7)

# ══════════════════════════════════════════════════════════════════════════════
# 10 SCÈNES
# ══════════════════════════════════════════════════════════════════════════════

# ── 01 Linear Regression ──────────────────────────────────────────────────────
def s01(ax, t, col, rng):
    style_ax(ax); ax.set_facecolor(BG)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_xlabel("Feature X"); ax.set_ylabel("Target y")
    title_band(ax,"01","Linear Regression","y = wX + b  →  minimize MSE = Σ(y - ŷ)²",col)

    np.random.seed(7)
    X = rng.uniform(1,9,40)
    y = 1.1*X + 1.5 + rng.normal(0,0.8,40)

    # Points qui apparaissent
    n = max(2, int(t*42)); n = min(n, 40)
    glow_pts(ax, X[:n], y[:n], col, s=28, a=0.65)

    # Droite de régression qui se trace
    if t > 0.30:
        al = ph(t, 0.30, 0.58)
        xf = np.linspace(0.5, 9.5, 200)
        ax.plot(xf, 1.1*xf+1.5, color=col, lw=2.5, alpha=al, zorder=5)

        # Résidus
        if t > 0.55:
            al2 = ph(t, 0.55, 0.78)
            for i in range(0, n, 4):
                yhat = 1.1*X[i]+1.5
                ax.plot([X[i],X[i]], [y[i],yhat],
                        color=RED, lw=1.0, alpha=al2*0.55, zorder=4)

    # MSE dynamique
    if t > 0.60:
        mse = np.mean((y[:n]-(1.1*X[:n]+1.5))**2)
        al3 = ph(t, 0.60, 0.80)
        ax.text(0.35, 9.2, f"MSE = {mse:.3f}", fontsize=9.5,
                color=col, fontfamily="monospace", fontweight="bold", alpha=al3,
                bbox=dict(boxstyle="round,pad=0.3",facecolor="white",
                          edgecolor=col, alpha=al3))

    if t > 0.45:
        code_snippet(ax, 0.3, 9.0,
            [("from sklearn.linear_model",  CODE_KW),
             ("  import LinearRegression",  CODE_KW),
             ("",                           CODE_FG),
             ("model = LinearRegression()", CODE_FN),
             ("model.fit(X_train, y_train)",CODE_FG),
             ("y_pred = model.predict(X)",  CODE_FG),
             ("# coef_: w,  intercept_: b", CODE_CMT),],
            w=4.8)

# ── 02 Logistic Regression ────────────────────────────────────────────────────
def s02(ax, t, col, rng):
    style_ax(ax); ax.set_facecolor(BG)
    ax.set_xlim(-5, 5); ax.set_ylim(-0.08, 1.12)
    ax.set_xlabel("z = wX + b"); ax.set_ylabel("σ(z) — probability")
    title_band(ax,"02","Logistic Regression","σ(z) = 1/(1+e⁻ᶻ)  →  Binary Classification",col)

    z  = np.linspace(-5, 5, 300)
    sig = 1/(1+np.exp(-z))

    # Courbe sigmoïde
    n = max(2, int(t*310)); n = min(n, 300)
    ax.plot(z[:n], sig[:n], color=col, lw=2.8, alpha=0.90, zorder=5)
    ax.fill_between(z[:n], sig[:n], color=col, alpha=0.09, zorder=2)

    # Lignes de référence
    if t > 0.30:
        al = ph(t, 0.30, 0.55)
        ax.axhline(0.5, color=MUTED, lw=1.0, ls="--", alpha=al*0.6)
        ax.axvline(0,   color=MUTED, lw=1.0, ls="--", alpha=al*0.6)
        ax.text(0.15, 0.52, "threshold = 0.5", fontsize=8,
                color=MUTED, fontfamily="monospace", alpha=al)

    # Points de données des deux classes
    if t > 0.40:
        al2 = ph(t, 0.40, 0.65)
        Xp = rng.uniform(1, 4.5, 15)
        Xn = rng.uniform(-4.5, -1, 15)
        ax.scatter(Xp, 1/(1+np.exp(-Xp)), s=35, color=GREEN,
                   alpha=al2*0.80, zorder=6, edgecolors="white", lw=0.7,
                   label="Class 1")
        ax.scatter(Xn, 1/(1+np.exp(-Xn)), s=35, color=RED,
                   alpha=al2*0.80, zorder=6, edgecolors="white", lw=0.7,
                   label="Class 0")
        ax.legend(facecolor="white", edgecolor=BORD, fontsize=8, loc="center right")

    # Formule
    if t > 0.55:
        al3 = ph(t, 0.55, 0.75)
        ax.text(1.2, 0.10, r"$\hat{y} = 1$ if $\sigma(z) \geq 0.5$",
                fontsize=9, color=GREEN, fontfamily="serif",
                alpha=al3, style="italic")
        ax.text(1.2, 0.02, r"$\hat{y} = 0$ if $\sigma(z) < 0.5$",
                fontsize=9, color=RED, fontfamily="serif",
                alpha=al3, style="italic")

    if t > 0.42:
        code_snippet(ax, -5.0, 1.10,
            [("from sklearn.linear_model",      CODE_KW),
             ("  import LogisticRegression",    CODE_KW),
             ("",                               CODE_FG),
             ("clf = LogisticRegression()",     CODE_FN),
             ("clf.fit(X_train, y_train)",      CODE_FG),
             ("proba = clf.predict_proba(X)",   CODE_FG),
             ("# proba[:,1] = P(class=1)",      CODE_CMT),],
            w=4.6)

# ── 03 Decision Tree ─────────────────────────────────────────────────────────
def s03(ax, t, col, rng):
    style_ax(ax); ax.set_facecolor(BG)
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.5)
    ax.axis("off"); ax.set_facecolor(BG)
    title_band(ax,"03","Decision Tree","Recursive splitting  →  minimize impurity (Gini/Entropy)",col)

    # Nœuds de l'arbre
    nodes = [
        # (x, y, label, col, cond)
        (5.0, 6.8, "X₁ ≤ 3.5 ?",          col,    "Root"),
        (2.5, 5.2, "X₂ ≤ 2.0 ?",          TEAL,   "Left branch"),
        (7.5, 5.2, "X₁ ≤ 6.0 ?",          ORANGE, "Right branch"),
        (1.2, 3.6, "Class A\n(Gini=0.00)", GREEN,  "Leaf"),
        (3.8, 3.6, "Class B\n(Gini=0.10)", RED,    "Leaf"),
        (6.2, 3.6, "Class B\n(Gini=0.05)", RED,    "Leaf"),
        (8.8, 3.6, "Class A\n(Gini=0.02)", GREEN,  "Leaf"),
    ]
    edges = [(0,1),(0,2),(1,3),(1,4),(2,5),(2,6)]
    coords = [(n[0],n[1]) for n in nodes]

    for ei, (src, dst) in enumerate(edges):
        a = ph(t, ei*0.07, ei*0.07+0.20)
        xs, ys = coords[src]; xd, yd = coords[dst]
        ax.plot([xs, xd], [ys-0.40, yd+0.40],
                color=BORD, lw=1.5, alpha=a, zorder=3)
        # Label Yes/No
        mid_x = (xs+xd)/2; mid_y = (ys+yd)/2
        lbl = "Yes" if dst%2==1 else "No"
        ax.text(mid_x+0.1, mid_y, lbl, fontsize=7.5, color=MUTED,
                fontfamily="monospace", alpha=a*0.8)

    for i, (x, y, label, c, tip) in enumerate(nodes):
        is_leaf = "Leaf" in tip
        a = ph(t, i*0.09, i*0.09+0.22)
        bw = 1.8 if is_leaf else 2.0; bh = 0.70 if is_leaf else 0.62
        rect = FancyBboxPatch((x-bw/2, y-bh/2), bw, bh,
                              boxstyle="round,pad=0.08",
                              facecolor=c+"18", edgecolor=c,
                              linewidth=1.8, alpha=a, zorder=5)
        ax.add_patch(rect)
        for li, line in enumerate(label.split("\n")):
            ax.text(x, y+(0.12 if "\n" in label else 0)-li*0.22,
                    line, fontsize=8.5 if not is_leaf else 8,
                    ha="center", va="center", color=c,
                    fontfamily="monospace", fontweight="bold",
                    alpha=a, zorder=6)

    if t > 0.50:
        code_snippet(ax, 0.3, 2.4,
            [("from sklearn.tree",         CODE_KW),
             ("  import DecisionTreeClassifier", CODE_KW),
             ("",                          CODE_FG),
             ("dt = DecisionTreeClassifier(", CODE_FN),
             ("  max_depth=4,",            CODE_NUM),
             ('  criterion="gini")',       CODE_STR),
             ("dt.fit(X_train, y_train)",  CODE_FG),],
            w=5.0)

# ── 04 K-Means Clustering ────────────────────────────────────────────────────
def s04(ax, t, col, rng):
    style_ax(ax); ax.set_facecolor(BG)
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5)
    ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
    title_band(ax,"04","K-Means Clustering","Assign  →  Update centroids  →  Repeat until convergence",col)

    # 3 clusters fixes
    centers_true  = np.array([[-2,2],[2,2],[0,-2.5]])
    centers_init  = np.array([[-3,3],[3,3],[0,-4]])
    cluster_cols  = [BLUE, ORANGE, GREEN]

    pts = []
    for ci, c_true in enumerate(centers_true):
        p = rng.multivariate_normal(c_true, [[0.5,0],[0,0.5]], 30)
        pts.append(p)

    # Évolution des centroides
    step = t * 8
    alpha_c = [0,0,0]
    for ci in range(3):
        frac = cl(step/1.5 - ci*0.5)
        centers_init[ci] = centers_init[ci]*(1-ease(frac)) + centers_true[ci]*ease(frac)

    # Points colorés par cluster (progressif)
    n_pts = max(2, int(t*32)); n_pts = min(n_pts, 30)
    for ci, (p, cc) in enumerate(zip(pts, cluster_cols)):
        glow_pts(ax, p[:n_pts,0], p[:n_pts,1], cc, s=22, a=0.60)

    # Centroides
    if t > 0.20:
        al = ph(t, 0.20, 0.40)
        for ci, (c, cc) in enumerate(zip(centers_init, cluster_cols)):
            ax.scatter(*c, s=220, color=cc, marker="*",
                       zorder=8, edgecolors="white", lw=1.5, alpha=al)
            ax.scatter(*c, s=500, color=cc, alpha=al*0.15, zorder=6)

    # Itération visible
    if t > 0.55:
        al2 = ph(t, 0.55, 0.75)
        it = int(t*5)
        ax.text(0.02, 0.08, f"Iteration {it} / 8",
                transform=ax.transAxes, fontsize=9,
                color=col, fontfamily="monospace", fontweight="bold",
                alpha=al2,
                bbox=dict(boxstyle="round,pad=0.3",facecolor="white",
                          edgecolor=col, alpha=al2))

    if t > 0.40:
        code_snippet(ax, -4.4, 4.4,
            [("from sklearn.cluster import KMeans", CODE_KW),
             ("",                                   CODE_FG),
             ("km = KMeans(",                       CODE_FN),
             ("  n_clusters=3,",                   CODE_NUM),
             ("  init='k-means++',",               CODE_STR),
             ("  n_init=10)",                      CODE_NUM),
             ("km.fit(X)",                         CODE_FG),
             ("labels = km.labels_",              CODE_FG),
             ("centers = km.cluster_centers_",    CODE_FG),],
            w=5.0)

# ── 05 SVM ───────────────────────────────────────────────────────────────────
def s05(ax, t, col, rng):
    style_ax(ax); ax.set_facecolor(BG)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
    title_band(ax,"05","Support Vector Machine","Maximize margin  →  optimal hyperplane",col)

    # Points deux classes
    rng2 = np.random.default_rng(12)
    Xp = rng2.multivariate_normal([2,2], [[0.5,0],[0,0.5]], 25)
    Xn = rng2.multivariate_normal([-2,-2],[[0.5,0],[0,0.5]], 25)

    n = max(2, int(t*27)); n=min(n,25)
    glow_pts(ax, Xp[:n,0], Xp[:n,1], BLUE, s=25, a=0.65)
    glow_pts(ax, Xn[:n,0], Xn[:n,1], ORANGE, s=25, a=0.65)

    # Hyperplan + marges
    if t > 0.30:
        al = ph(t, 0.30, 0.55)
        xf = np.linspace(-4, 4, 100)
        ax.plot(xf, -xf, color=col, lw=2.5, alpha=al, zorder=5)  # frontière
        ax.plot(xf, -xf+1.5, color=col, lw=1.2, ls="--", alpha=al*0.6)  # marge +
        ax.plot(xf, -xf-1.5, color=col, lw=1.2, ls="--", alpha=al*0.6)  # marge -
        ax.fill_between(xf, -xf-1.5, -xf+1.5, color=col, alpha=al*0.07)

    # Vecteurs de support
    if t > 0.55:
        al2 = ph(t, 0.55, 0.75)
        svs = np.array([[1.2,0.9],[0.8,1.3],[-1.1,-0.8],[-0.9,-1.2]])
        ax.scatter(svs[:,0], svs[:,1], s=120, color="none",
                   edgecolors=AMBER, lw=2.0, zorder=7, alpha=al2)
        ax.text(0.35, 3.5, "margin = 2/‖w‖", fontsize=9,
                color=col, fontfamily="monospace",
                fontweight="bold", alpha=al2,
                bbox=dict(boxstyle="round,pad=0.3",facecolor="white",
                          edgecolor=col,alpha=al2))
        ax.annotate("Support\nVectors", xy=(1.0,1.1), xytext=(2.5,2.5),
                    fontsize=8, color=AMBER, fontfamily="monospace",
                    arrowprops=dict(arrowstyle="->",color=AMBER,lw=1.2),
                    alpha=al2)

    if t > 0.40:
        code_snippet(ax, -4.0, 4.0,
            [("from sklearn.svm import SVC",      CODE_KW),
             ("",                                  CODE_FG),
             ("svm = SVC(",                        CODE_FN),
             ('  kernel="rbf",',                  CODE_STR),
             ("  C=1.0, gamma='scale')",          CODE_NUM),
             ("svm.fit(X_train, y_train)",         CODE_FG),
             ("# n_support_: nb support vectors", CODE_CMT),],
            w=4.8)

# ── 06 Random Forest ─────────────────────────────────────────────────────────
def s06(ax, t, col, rng):
    style_ax(ax); ax.set_facecolor(BG)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
    title_band(ax,"06","Random Forest","Bagging  →  N decision trees  →  majority vote",col)

    rng2 = np.random.default_rng(5)
    X = rng2.multivariate_normal([0,0],[[2,0.5],[0.5,2]],200)
    y = (X[:,0]+X[:,1]>0).astype(int)

    # Frontière complexe
    xx, yy = np.meshgrid(np.linspace(-4,4,120), np.linspace(-4,4,120))
    Z = np.sin(xx)*np.cos(yy) + 0.3*(xx+yy>0).astype(float)
    Z = (Z > 0.1).astype(float)

    if t > 0.25:
        al = ph(t, 0.25, 0.50)
        cmap = LinearSegmentedColormap.from_list("rf", [BLUE+"40", ORANGE+"40"])
        ax.contourf(xx, yy, Z, levels=1, cmap=cmap, alpha=al*0.55, zorder=1)
        ax.contour(xx, yy, Z, levels=1, colors=[col], linewidths=2.0,
                   alpha=al*0.80, zorder=4)

    # Points
    n = max(2, int(t*210)); n=min(n,200)
    glow_pts(ax, X[:n,0][y[:n]==1], X[:n,1][y[:n]==1], BLUE, s=16, a=0.50)
    glow_pts(ax, X[:n,0][y[:n]==0], X[:n,1][y[:n]==0], ORANGE, s=16, a=0.50)

    # Arbres individuels (icônes)
    if t > 0.55:
        al2 = ph(t, 0.55, 0.75)
        n_trees = int(t*8)+2
        for ti in range(min(n_trees, 8)):
            tx = -3.8 + ti*0.95
            ax.text(tx, -3.5, "🌲", fontsize=14, alpha=al2*0.75, ha="center")
        ax.text(0.0, -4.0, f"{min(n_trees,8)}/{8} trees voted",
                fontsize=8.5, ha="center", color=col,
                fontfamily="monospace", alpha=al2)

    if t > 0.38:
        code_snippet(ax, -4.0, 4.0,
            [("from sklearn.ensemble",          CODE_KW),
             ("  import RandomForestClassifier",CODE_KW),
             ("",                               CODE_FG),
             ("rf = RandomForestClassifier(",   CODE_FN),
             ("  n_estimators=100,",           CODE_NUM),
             ("  max_depth=5,",               CODE_NUM),
             ("  random_state=42)",           CODE_NUM),
             ("rf.fit(X_train, y_train)",      CODE_FG),],
            w=5.2)

# ── 07 Gradient Boosting ─────────────────────────────────────────────────────
def s07(ax, t, col, rng):
    style_ax(ax); ax.set_facecolor(BG)
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Boosting rounds (n_estimators)")
    ax.set_ylabel("Score")
    title_band(ax,"07","Gradient Boosting","Sequential trees  →  each corrects previous errors",col)

    x_steps = np.arange(1, 101)
    rng2 = np.random.default_rng(3)
    # Train score (croît vite, puis plateau)
    train_score = 1 - 0.68*np.exp(-x_steps/18) + rng2.normal(0,0.008,100)
    train_score = np.clip(train_score, 0.3, 1.0)
    # Val score (croît, puis légère surapprentissage)
    val_score   = 1 - 0.72*np.exp(-x_steps/22) - x_steps/3000 + rng2.normal(0,0.009,100)
    val_score   = np.clip(val_score, 0.3, 1.0)

    n = max(2, int(t*105)); n=min(n,100)
    ax.plot(x_steps[:n], train_score[:n], color=BLUE, lw=2.2,
            label="Train score", alpha=0.88)
    ax.plot(x_steps[:n], val_score[:n], color=col, lw=2.2,
            linestyle="--", label="Val score", alpha=0.88)

    # Meilleure epoch
    best = np.argmax(val_score[:n])
    if t > 0.55:
        al = ph(t, 0.55, 0.75)
        ax.axvline(x_steps[best], color=GREEN, lw=1.5, ls=":", alpha=al*0.7)
        ax.scatter([x_steps[best]], [val_score[best]], s=80, color=GREEN,
                   zorder=8, alpha=al, edgecolors="white", lw=1.2)
        ax.text(x_steps[best]+1, 0.55, f"best round: {x_steps[best]}",
                fontsize=8, color=GREEN, fontfamily="monospace", alpha=al)

    # Zone surapprentissage
    if t > 0.65:
        al2 = ph(t, 0.65, 0.82)
        ax.axvspan(70, 100, color=RED, alpha=al2*0.06)
        ax.text(75, 0.72, "overfitting\nrisk", fontsize=8,
                color=RED, fontfamily="monospace", alpha=al2, ha="center")

    if t > 0.30:
        ax.legend(facecolor="white", edgecolor=BORD, fontsize=8.5)

    if t > 0.42:
        code_snippet(ax, 2, 1.04,
            [("from sklearn.ensemble",        CODE_KW),
             ("  import GradientBoostingClassifier", CODE_KW),
             ("# or: pip install xgboost",   CODE_CMT),
             ("import xgboost as xgb",       CODE_KW),
             ("",                            CODE_FG),
             ("xgb_clf = xgb.XGBClassifier(",CODE_FN),
             ("  n_estimators=100,",         CODE_NUM),
             ("  learning_rate=0.05,",       CODE_NUM),
             ("  max_depth=6)",              CODE_NUM),],
            w=5.2)

# ── 08 Principal Component Analysis ──────────────────────────────────────────
def s08(ax, t, col, rng):
    style_ax(ax); ax.set_facecolor(BG)
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5)
    ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
    title_band(ax,"08","PCA — Principal Component Analysis","Maximize variance  →  reduce dimensions",col)

    rng2 = np.random.default_rng(9)
    angle = np.radians(35)
    cov   = np.array([[3,2],[2,2]])
    X     = rng2.multivariate_normal([0,0], cov, 150)

    n = max(2, int(t*160)); n=min(n,150)
    glow_pts(ax, X[:n,0], X[:n,1], col, s=18, a=0.45)

    # Axes PCA
    if t > 0.30:
        al = ph(t, 0.30, 0.55)
        pc1 = np.array([np.cos(angle), np.sin(angle)])*3.2
        pc2 = np.array([-np.sin(angle), np.cos(angle)])*1.8
        ax.annotate("", xy=pc1, xytext=-pc1,
                    arrowprops=dict(arrowstyle="->",color=BLUE,lw=2.2,alpha=al))
        ax.annotate("", xy=pc2, xytext=-pc2,
                    arrowprops=dict(arrowstyle="->",color=ORANGE,lw=2.2,alpha=al))
        ax.text(pc1[0]+0.2, pc1[1]+0.2, "PC1\n(max variance)",
                fontsize=8.5, color=BLUE, fontfamily="monospace",
                fontweight="bold", alpha=al)
        ax.text(pc2[0]+0.2, pc2[1]+0.2, "PC2",
                fontsize=8.5, color=ORANGE, fontfamily="monospace",
                fontweight="bold", alpha=al)

    # Projection sur PC1
    if t > 0.58:
        al2 = ph(t, 0.58, 0.78)
        v1  = np.array([np.cos(angle), np.sin(angle)])
        for i in range(0, min(n, 20)):
            proj_s  = X[i] @ v1
            proj_pt = proj_s * v1
            ax.plot([X[i,0], proj_pt[0]], [X[i,1], proj_pt[1]],
                    color=MUTED, lw=0.6, alpha=al2*0.5)

    # Variance expliquée
    if t > 0.62:
        al3 = ph(t, 0.62, 0.80)
        ax.text(0.02, 0.08, "PC1 explains 82% variance",
                transform=ax.transAxes, fontsize=9,
                color=col, fontfamily="monospace", fontweight="bold",
                alpha=al3,
                bbox=dict(boxstyle="round,pad=0.3",facecolor="white",
                          edgecolor=col,alpha=al3))

    if t > 0.40:
        code_snippet(ax, -4.4, 4.4,
            [("from sklearn.decomposition import PCA", CODE_KW),
             ("",                                       CODE_FG),
             ("pca = PCA(n_components=2)",             CODE_FN),
             ("pca.fit(X_train)",                      CODE_FG),
             ("X_reduced = pca.transform(X)",          CODE_FG),
             ("",                                       CODE_FG),
             ("# explained variance ratio:",           CODE_CMT),
             ("pca.explained_variance_ratio_",         CODE_FG),
             ("# → [0.82, 0.14]",                     CODE_CMT),],
            w=5.0)

# ── 09 Neural Network ────────────────────────────────────────────────────────
def s09(ax, t, col, rng):
    style_ax(ax); ax.set_facecolor(BG)
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.5)
    ax.axis("off"); ax.set_facecolor(BG)
    title_band(ax,"09","Neural Network  (MLP)","Forward pass  →  Loss  →  Backpropagation  →  Update",col)

    layers = [
        (1.2, [2.5, 3.5, 4.5, 5.0, 5.5], "Input", INK2),
        (3.5, [2.0, 3.0, 4.0, 5.0, 6.0], "Hidden 1", BLUE),
        (5.8, [2.5, 3.5, 4.5, 5.5],       "Hidden 2", VIOLET),
        (8.0, [3.5, 4.5],                  "Output", col),
    ]

    # Connexions
    for li in range(len(layers)-1):
        lx, lnodes, _, _ = layers[li]
        rx, rnodes, _, _ = layers[li+1]
        for yn in lnodes:
            for yr in rnodes:
                a = ph(t, li*0.12, li*0.12+0.25)
                ax.plot([lx+0.18, rx-0.18], [yn, yr],
                        color=BORD, lw=0.55, alpha=a*0.55, zorder=2)

    # Nœuds
    for li, (lx, lnodes, lname, lc) in enumerate(layers):
        al = ph(t, li*0.12+0.05, li*0.12+0.30)
        for yn in lnodes:
            circ = plt.Circle((lx, yn), 0.22,
                              facecolor=lc+"20", edgecolor=lc,
                              linewidth=1.8, alpha=al, zorder=5)
            ax.add_patch(circ)
        ax.text(lx, 1.4, lname, fontsize=8.5, ha="center",
                color=lc, fontfamily="monospace", fontweight="bold",
                alpha=al)

    # Flux forward pass
    if t > 0.55:
        al2 = ph(t, 0.55, 0.75)
        for li in range(len(layers)-1):
            lx = layers[li][0]; rx = layers[li+1][0]
            mid = (lx+rx)/2
            ax.annotate("", xy=(rx-0.25, 4.0), xytext=(lx+0.25, 4.0),
                        arrowprops=dict(arrowstyle="->",color=col,
                                       lw=1.8,alpha=al2))
        ax.text(4.6, 4.35, "forward →", fontsize=8,
                color=col, fontfamily="monospace",
                alpha=al2, ha="center")

    # Backprop
    if t > 0.72:
        al3 = ph(t, 0.72, 0.90)
        ax.annotate("", xy=(1.5, 3.2), xytext=(7.8, 3.2),
                    arrowprops=dict(arrowstyle="->",color=RED,
                                   lw=1.5,linestyle="dashed",alpha=al3))
        ax.text(4.6, 2.85, "← backprop  (∂L/∂w)",
                fontsize=8, color=RED, ha="center",
                fontfamily="monospace", alpha=al3)

    if t > 0.40:
        code_snippet(ax, 0.3, 2.2,
            [("import torch.nn as nn",          CODE_KW),
             ("",                                CODE_FG),
             ("model = nn.Sequential(",         CODE_FN),
             ("  nn.Linear(5, 64),",            CODE_NUM),
             ("  nn.ReLU(),",                   CODE_FG),
             ("  nn.Linear(64, 32),",           CODE_NUM),
             ("  nn.ReLU(),",                   CODE_FG),
             ("  nn.Linear(32, 2))",            CODE_NUM),],
            w=4.4)

# ── 10 Cross-Validation & Model Selection ─────────────────────────────────────
def s10(ax, t, col, rng):
    style_ax(ax); ax.set_facecolor(BG)
    ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    ax.axis("off"); ax.set_facecolor(BG)
    title_band(ax,"10","Cross-Validation & Model Selection","k-Fold CV  →  unbiased generalization estimate",col)

    K = 5
    fold_h = 0.65
    fold_gap = 0.18
    fold_y_start = 5.8
    fold_w = 7.5

    # k folds
    for k in range(K):
        a = ph(t, k*0.10, k*0.10+0.22)
        fy = fold_y_start - k*(fold_h+fold_gap)

        # Train portions (divisé en K segments)
        seg_w = fold_w / K
        for seg in range(K):
            is_val = (seg == k)
            fc = col if is_val else BORD+"80"
            ec = col if is_val else BORD
            rect = FancyBboxPatch((1.2+seg*seg_w, fy), seg_w-0.06, fold_h,
                                  boxstyle="square,pad=0",
                                  facecolor=fc, edgecolor=ec,
                                  linewidth=1.2, alpha=a, zorder=5)
            ax.add_patch(rect)
            if is_val:
                ax.text(1.2+seg*seg_w+seg_w/2, fy+fold_h/2,
                        "VAL", fontsize=8, ha="center", va="center",
                        color="white", fontfamily="monospace",
                        fontweight="bold", alpha=a, zorder=6)
            else:
                ax.text(1.2+seg*seg_w+seg_w/2, fy+fold_h/2,
                        "TR", fontsize=7.5, ha="center", va="center",
                        color=INK2, fontfamily="monospace",
                        alpha=a*0.6, zorder=6)

        # Score du fold
        score = 0.85 + rng.uniform(-0.04, 0.04)
        ax.text(9.0, fy+fold_h/2, f"{score:.3f}",
                fontsize=9, va="center", color=col,
                fontfamily="monospace", fontweight="bold",
                alpha=a, zorder=6)
        ax.text(0.3, fy+fold_h/2, f"Fold {k+1}",
                fontsize=8.5, va="center", color=INK2,
                fontfamily="monospace", alpha=a, zorder=6)

    # Moyenne et std
    if t > 0.60:
        al = ph(t, 0.60, 0.80)
        ax.text(3.5, 2.4,
                "Mean CV score: 0.852 ± 0.018",
                fontsize=10, ha="center", color=col,
                fontfamily="monospace", fontweight="bold",
                alpha=al,
                bbox=dict(boxstyle="round,pad=0.4",facecolor="white",
                          edgecolor=col, alpha=al))

    if t > 0.42:
        code_snippet(ax, 0.4, 1.8,
            [("from sklearn.model_selection",         CODE_KW),
             ("  import cross_val_score",             CODE_KW),
             ("",                                      CODE_FG),
             ("scores = cross_val_score(",            CODE_FN),
             ("  model, X, y,",                      CODE_FG),
             ("  cv=5,",                             CODE_NUM),
             ('  scoring="accuracy")',               CODE_STR),
             ("print(f\"{scores.mean():.3f}\"",      CODE_FG),
             ("      f\" ± {scores.std():.3f}\")",   CODE_FG),],
            w=5.2)

# ── Dispatch ───────────────────────────────────────────────────────────────────
SCENES = [s01, s02, s03, s04, s05, s06, s07, s08, s09, s10]

# ── Animation ──────────────────────────────────────────────────────────────────
def make_gif():
    timeline = []
    for si in range(TOTAL):
        for fi in range(STAGE_FRAMES): timeline.append((si, fi))
        for _  in range(PAUSE_FRAMES): timeline.append((si, STAGE_FRAMES-1))

    total = len(timeline)
    print(f"Frames : {total}  ({total/FPS:.1f}s @ {FPS}fps)")

    rngs = [np.random.default_rng(42+i) for i in range(TOTAL)]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=DPI)
    fig.patch.set_facecolor(BG)

    fig.text(0.5, 0.006, FOOTER, ha="center", va="bottom",
             color=MUTED, fontsize=6.2, fontfamily="monospace")

    fa = []

    def rm():
        for a_ in fa:
            try: a_.remove()
            except: pass
        fa.clear()

    def update(fi):
        nonlocal fa
        rm()
        si, lf = timeline[fi]
        t_ = ease(lf / max(1, STAGE_FRAMES-1))
        ax.cla()
        ax.set_facecolor(BG)
        SCENES[si](ax, t_, SCENE_COLS[si], rngs[si])
        return []

    anim = FuncAnimation(fig, update, frames=total,
                         interval=1000//FPS, blit=False)
    print("Génération...")
    anim.save(OUT, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    sz = os.path.getsize(OUT)/1024/1024
    print(f"{OUT}  —  {total/FPS:.1f}s  —  {sz:.1f} MB")

if __name__ == "__main__":
    make_gif()
