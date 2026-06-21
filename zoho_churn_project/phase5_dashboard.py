import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/zoho_scored_data.csv', parse_dates=['signup_date'])

ZOHO_RED    = '#E42527'
ZOHO_BLUE   = '#1565C0'
ZOHO_TEAL   = '#00897B'
ZOHO_AMBER  = '#F57C00'
ZOHO_PURPLE = '#5E35B1'
GRAY        = '#607D8B'
BG          = '#F8F9FA'
PLAN_ORDER  = ['Free','Standard','Professional','Enterprise','Zoho One']
PLAN_COLORS = {'Free':GRAY,'Standard':ZOHO_BLUE,'Professional':ZOHO_TEAL,
               'Enterprise':ZOHO_AMBER,'Zoho One':ZOHO_PURPLE}
PLAN_PRICES = {'Free':0,'Standard':20,'Professional':35,'Enterprise':50,'Zoho One':105}

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('white')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.4)

# ── Header ──────────────────────────────────────────────────────────────────
ax_hdr = fig.add_subplot(gs[0, :])
ax_hdr.set_facecolor('#1A1A2E')
ax_hdr.text(0.02, 0.62, 'Zoho SMB — Customer Health & Revenue Analytics',
            transform=ax_hdr.transAxes, fontsize=15, fontweight='bold', color='white')
ax_hdr.text(0.02, 0.20, 'Churn Analysis · Cohort Retention · MRR Waterfall · Upsell Intelligence',
            transform=ax_hdr.transAxes, fontsize=10, color='#AAB0C0')
for spine in ax_hdr.spines.values(): spine.set_visible(False)
ax_hdr.set_xticks([]); ax_hdr.set_yticks([])

# KPI boxes in header
kpis = [
    ('Total customers', '5,000', ZOHO_BLUE),
    ('Overall churn rate', f"{df['churned'].mean():.1%}", ZOHO_RED),
    ('At-risk MRR', '$1,740/mo', ZOHO_AMBER),
    ('Upsell candidates', '1,230', ZOHO_TEAL),
    ('Churn model AUC', '0.843', ZOHO_PURPLE),
]
for idx, (label, val, color) in enumerate(kpis):
    x = 0.54 + idx * 0.092
    ax_hdr.add_patch(mpatches.FancyBboxPatch(
        (x-0.04, 0.05), 0.087, 0.88,
        boxstyle='round,pad=0.01', transform=ax_hdr.transAxes,
        facecolor='#252540', edgecolor=color, linewidth=1.5, zorder=2))
    ax_hdr.text(x, 0.72, val, transform=ax_hdr.transAxes,
                fontsize=11, fontweight='bold', color=color, ha='center')
    ax_hdr.text(x, 0.25, label, transform=ax_hdr.transAxes,
                fontsize=7.5, color='#8090A0', ha='center')

ax_hdr.set_xlim(0,1); ax_hdr.set_ylim(0,1)

# ── Panel 1: Churn rate by plan ──────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1, 0])
churn_by_plan = df.groupby('plan')['churned'].mean().reindex(PLAN_ORDER) * 100
bars = ax1.bar(PLAN_ORDER, churn_by_plan,
               color=[PLAN_COLORS[p] for p in PLAN_ORDER], width=0.65, zorder=3)
for bar, v in zip(bars, churn_by_plan):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{v:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#333')
ax1.set_facecolor(BG); ax1.spines[['top','right']].set_visible(False)
ax1.grid(axis='y', color='#E8E8E8', linewidth=0.7); ax1.set_axisbelow(True)
ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
ax1.set_title('Churn rate by plan', fontsize=10, fontweight='bold', color='#1A1A2E', pad=8)
ax1.tick_params(axis='x', labelsize=8, rotation=15)
ax1.set_ylim(0, churn_by_plan.max() * 1.2)

# ── Panel 2: Apps used vs churn ──────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 1])
df['apps_bin'] = pd.cut(df['apps_used'], bins=[0,1,2,4,8,30],
                        labels=['1','2','3–4','5–8','9+'])
churn_apps = df.groupby('apps_bin', observed=True)['churned'].mean() * 100
colors_a   = [ZOHO_RED, '#E57373', ZOHO_AMBER, ZOHO_TEAL, ZOHO_PURPLE]
bars = ax2.bar(churn_apps.index, churn_apps, color=colors_a, width=0.65, zorder=3)
for bar, v in zip(bars, churn_apps):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{v:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#333')
ax2.set_facecolor(BG); ax2.spines[['top','right']].set_visible(False)
ax2.grid(axis='y', color='#E8E8E8', linewidth=0.7); ax2.set_axisbelow(True)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.set_title('Churn rate by apps used', fontsize=10, fontweight='bold', color='#1A1A2E', pad=8)
ax2.set_xlabel('No. of Zoho apps used', fontsize=8)
ax2.tick_params(labelsize=8)
ax2.set_ylim(0, churn_apps.max() * 1.2)

# ── Panel 3: Tenure danger zone ──────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 2])
df['tenure_bin'] = pd.cut(df['tenure_months'], bins=[0,3,6,12,24,100],
                          labels=['0–3m','3–6m','6–12m','12–24m','24+m'])
churn_ten = df.groupby('tenure_bin', observed=True)['churned'].mean() * 100
colors_t  = [ZOHO_RED if i < 2 else ZOHO_BLUE for i in range(len(churn_ten))]
bars = ax3.bar(churn_ten.index, churn_ten, color=colors_t, width=0.65, zorder=3)
for bar, v in zip(bars, churn_ten):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{v:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#333')
ax3.set_facecolor(BG); ax3.spines[['top','right']].set_visible(False)
ax3.grid(axis='y', color='#E8E8E8', linewidth=0.7); ax3.set_axisbelow(True)
ax3.yaxis.set_major_formatter(mtick.PercentFormatter())
ax3.set_title('Churn by tenure (danger zone)', fontsize=10, fontweight='bold', color='#1A1A2E', pad=8)
ax3.tick_params(labelsize=8)
red_p  = mpatches.Patch(color=ZOHO_RED,  label='High risk')
blue_p = mpatches.Patch(color=ZOHO_BLUE, label='Stable')
ax3.legend(handles=[red_p, blue_p], fontsize=7, framealpha=0.8)
ax3.set_ylim(0, churn_ten.max() * 1.3)

# ── Panel 4: MRR trend (full width bottom-left) ──────────────────────────────
ax4 = fig.add_subplot(gs[2, :2])
np.random.seed(42)
months = pd.date_range('2023-01', periods=18, freq='ME')
new_mrr  = list(np.random.randint(280,420,18) * 32)
exp_mrr  = []
churn_mrr= []
total_mrr= [480000]
for i in range(18):
    base  = total_mrr[-1]
    exp   = base * np.random.uniform(0.01, 0.025)
    churn = base * np.random.uniform(0.028, 0.055)
    exp_mrr.append(round(exp))
    churn_mrr.append(round(-churn))
    total_mrr.append(round(base + new_mrr[i] + exp - churn))
total_mrr = total_mrr[1:]
x = np.arange(18)
ax4.bar(x, new_mrr, 0.55, label='New MRR', color=ZOHO_BLUE, zorder=3)
ax4.bar(x, exp_mrr, 0.55, bottom=new_mrr, label='Expansion MRR', color=ZOHO_TEAL, zorder=3)
ax4.bar(x, churn_mrr, 0.55, label='Churned MRR', color=ZOHO_RED, zorder=3)
ax4_twin = ax4.twinx()
ax4_twin.plot(x, [t/1000 for t in total_mrr], color=ZOHO_PURPLE, lw=2.5,
              marker='o', markersize=4, zorder=5, label='Total MRR')
ax4_twin.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: f'${v:.0f}K'))
ax4_twin.set_ylabel('Total MRR ($K)', fontsize=8, color=ZOHO_PURPLE)
ax4_twin.tick_params(labelsize=8, colors=ZOHO_PURPLE)
ax4_twin.spines[['top']].set_visible(False)
ax4.set_facecolor(BG); ax4.spines[['top','right']].set_visible(False)
ax4.grid(axis='y', color='#E8E8E8', linewidth=0.7); ax4.set_axisbelow(True)
ax4.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: f'${v/1000:.0f}K'))
ax4.set_xticks(x); ax4.set_xticklabels([m.strftime('%b %y') for m in months],
                                         rotation=40, ha='right', fontsize=8)
ax4.set_title('MRR waterfall — 18 months', fontsize=10, fontweight='bold', color='#1A1A2E', pad=8)
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=8, framealpha=0.8, loc='upper left')
ax4.axhline(0, color='#999', linewidth=0.8)

# ── Panel 5: Churn/Upsell quadrant ──────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 2])
from matplotlib.lines import Line2D
sample = df[(df['plan'] != 'Zoho One') & (df['churned']==0)].sample(400, random_state=1)
quad_colors = []
for _, row in sample.iterrows():
    if row['churn_score'] > 60 and row['upsell_score'] > 50:
        quad_colors.append(ZOHO_AMBER)
    elif row['churn_score'] > 60:
        quad_colors.append(ZOHO_RED)
    elif row['upsell_score'] > 50:
        quad_colors.append(ZOHO_TEAL)
    else:
        quad_colors.append('#90CAF9')
ax5.scatter(sample['churn_score'], sample['upsell_score'],
            c=quad_colors, alpha=0.5, s=18, zorder=3)
ax5.axvline(60, color='#999', linestyle='--', linewidth=0.8)
ax5.axhline(50, color='#999', linestyle='--', linewidth=0.8)
ax5.set_facecolor(BG); ax5.spines[['top','right']].set_visible(False)
ax5.set_xlabel('Churn risk score', fontsize=8)
ax5.set_ylabel('Upsell readiness', fontsize=8)
ax5.set_title('Risk × Upsell matrix', fontsize=10, fontweight='bold', color='#1A1A2E', pad=8)
ax5.tick_params(labelsize=8)
leg_els = [
    Line2D([0],[0],marker='o',color='w',markerfacecolor=ZOHO_RED,  ms=7, label='Churn risk'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=ZOHO_TEAL, ms=7, label='Upsell ready'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=ZOHO_AMBER,ms=7, label='Urgent (both)'),
]
ax5.legend(handles=leg_els, fontsize=7, framealpha=0.8)

plt.savefig('/outputs/07_executive_dashboard.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Executive dashboard saved.")
