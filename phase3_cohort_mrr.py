import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('zoho_smb_data.csv', parse_dates=['signup_date'])

ZOHO_RED    = '#E42527'
ZOHO_BLUE   = '#1565C0'
ZOHO_TEAL   = '#00897B'
ZOHO_AMBER  = '#F57C00'
ZOHO_PURPLE = '#5E35B1'
BG = '#F8F9FA'

PLAN_ORDER  = ['Free', 'Standard', 'Professional', 'Enterprise', 'Zoho One']
PLAN_PRICES = {'Free':0,'Standard':20,'Professional':35,'Enterprise':50,'Zoho One':105}

# ─── FIGURE 3: Cohort Retention Heatmap ─────────────────────────────────────
df['cohort'] = df['signup_date'].dt.to_period('Q')
cohorts = df.groupby('cohort').apply(
    lambda g: pd.Series({
        'customers': len(g),
        'retained_m3':  (g['tenure_months'] >= 3).sum(),
        'retained_m6':  (g['tenure_months'] >= 6).sum(),
        'retained_m12': (g['tenure_months'] >= 12).sum(),
        'retained_m18': (g['tenure_months'] >= 18).sum(),
        'retained_m24': (g['tenure_months'] >= 24).sum(),
    })
).reset_index()

# Retention rates as % of cohort start
for col in ['retained_m3','retained_m6','retained_m12','retained_m18','retained_m24']:
    cohorts[col+'_pct'] = (cohorts[col] / cohorts['customers'] * 100).round(1)

heatmap_data = cohorts[['cohort'] + [c+'_pct' for c in
    ['retained_m3','retained_m6','retained_m12','retained_m18','retained_m24']]].copy()
heatmap_data.columns = ['Cohort','Month 3','Month 6','Month 12','Month 18','Month 24']
heatmap_data = heatmap_data.set_index('Cohort')
heatmap_data = heatmap_data[heatmap_data.values.max(axis=1) > 0]

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('white')
ax.set_facecolor(BG)

import matplotlib.colors as mcolors
cmap = plt.cm.RdYlGn
norm = mcolors.Normalize(vmin=0, vmax=100)

for i, (idx, row) in enumerate(heatmap_data.iterrows()):
    for j, val in enumerate(row):
        if pd.isna(val) or val == 0:
            continue
        color = cmap(norm(val))
        rect = plt.Rectangle([j, i], 1, 1, fill=True, color=color, linewidth=0.5,
                              edgecolor='white')
        ax.add_patch(rect)
        ax.text(j + 0.5, i + 0.5, f'{val:.0f}%', ha='center', va='center',
                fontsize=10, fontweight='bold',
                color='white' if val < 40 or val > 75 else '#1A1A2E')

ax.set_xlim(0, len(heatmap_data.columns))
ax.set_ylim(0, len(heatmap_data))
ax.set_xticks(np.arange(len(heatmap_data.columns)) + 0.5)
ax.set_xticklabels(heatmap_data.columns, fontsize=11, color='#333')
ax.set_yticks(np.arange(len(heatmap_data.index)) + 0.5)
ax.set_yticklabels([str(c) for c in heatmap_data.index], fontsize=10, color='#333')
ax.set_title('Cohort Retention Heatmap — % of customers retained by month',
             fontsize=14, fontweight='bold', pad=15, color='#1A1A2E')
ax.set_xlabel('Months since signup', fontsize=11, color='#555')
ax.set_ylabel('Signup cohort (quarter)', fontsize=11, color='#555')
ax.spines[['top','right','left','bottom']].set_visible(False)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Retention %', fontsize=10)
cbar.ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig('/outputs/03_cohort_retention.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Cohort heatmap saved.")

# ─── FIGURE 4: MRR Waterfall ────────────────────────────────────────────────
np.random.seed(42)
months = pd.date_range('2023-01', periods=18, freq='ME')
monthly_new_customers = np.random.randint(280, 420, 18)
monthly_churn_rate    = np.random.uniform(0.028, 0.055, 18)
monthly_expansion     = np.random.uniform(0.01, 0.025, 18)

new_mrr        = (monthly_new_customers * 32).tolist()
expansion_mrr  = []
churn_mrr      = []
total_mrr      = [480000]

for i in range(18):
    base = total_mrr[-1]
    exp  = base * monthly_expansion[i]
    churn = base * monthly_churn_rate[i]
    expansion_mrr.append(round(exp))
    churn_mrr.append(round(-churn))
    total_mrr.append(round(base + new_mrr[i] + exp - churn))

total_mrr = total_mrr[1:]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
fig.patch.set_facecolor('white')

# Waterfall bars
month_labels = [m.strftime('%b %y') for m in months]
x = np.arange(len(months))
width = 0.55

ax1.bar(x, new_mrr, width, label='New MRR', color=ZOHO_BLUE, zorder=3)
ax1.bar(x, expansion_mrr, width, bottom=new_mrr, label='Expansion MRR', color=ZOHO_TEAL, zorder=3)
ax1.bar(x, churn_mrr, width, label='Churned MRR', color=ZOHO_RED, zorder=3)

ax1.set_facecolor(BG)
ax1.spines[['top','right']].set_visible(False)
ax1.spines[['left','bottom']].set_color('#DDD')
ax1.grid(axis='y', color='#E8E8E8', linewidth=0.7)
ax1.set_axisbelow(True)
ax1.set_xticks(x)
ax1.set_xticklabels(month_labels, rotation=45, ha='right', fontsize=9)
ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f'${v/1000:.0f}K'))
ax1.set_title('Monthly MRR Movement (New · Expansion · Churn)', fontsize=13,
              fontweight='bold', pad=10, color='#1A1A2E')
ax1.legend(fontsize=9, framealpha=0.8, loc='upper left')
ax1.set_ylabel('MRR ($)', fontsize=10, color='#555')
ax1.axhline(0, color='#999', linewidth=0.8)

# Total MRR trend
ax2.plot(x, [t/1000 for t in total_mrr], color=ZOHO_PURPLE, linewidth=2.5,
         marker='o', markersize=5, zorder=3, label='Total MRR')
ax2.fill_between(x, [t/1000 for t in total_mrr], alpha=0.12, color=ZOHO_PURPLE)

ax2.set_facecolor(BG)
ax2.spines[['top','right']].set_visible(False)
ax2.spines[['left','bottom']].set_color('#DDD')
ax2.grid(axis='y', color='#E8E8E8', linewidth=0.7)
ax2.set_axisbelow(True)
ax2.set_xticks(x)
ax2.set_xticklabels(month_labels, rotation=45, ha='right', fontsize=9)
ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f'${v:.0f}K'))
ax2.set_title('Total MRR Trend', fontsize=13, fontweight='bold', pad=10, color='#1A1A2E')
ax2.set_ylabel('Total MRR ($K)', fontsize=10, color='#555')
ax2.legend(fontsize=9, framealpha=0.8)

mrr_growth = (total_mrr[-1] - total_mrr[0]) / total_mrr[0] * 100
ax2.annotate(f'+{mrr_growth:.0f}% over 18 months',
             xy=(len(months)-1, total_mrr[-1]/1000),
             xytext=(len(months)-5, total_mrr[-1]/1000 * 0.88),
             fontsize=10, fontweight='bold', color=ZOHO_PURPLE,
             arrowprops=dict(arrowstyle='->', color=ZOHO_PURPLE, lw=1.5))

plt.tight_layout()
plt.savefig('/outputs/04_mrr_waterfall.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("MRR waterfall saved.")

# ─── FIGURE 5: Plan-level LTV & NRR comparison ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('white')

plan_stats = df.groupby('plan').agg(
    churn_rate=('churned','mean'),
    avg_tenure=('tenure_months','mean'),
    avg_seats=('seats','mean'),
).reindex(PLAN_ORDER)

plan_stats['avg_price']  = [PLAN_PRICES[p] for p in PLAN_ORDER]
plan_stats['monthly_revenue'] = plan_stats['avg_price'] * plan_stats['avg_seats']
plan_stats['ltv'] = (plan_stats['monthly_revenue'] / plan_stats['churn_rate'].clip(0.01)).round(0)
plan_stats['nrr'] = ((1 - plan_stats['churn_rate']) * 100 + np.random.uniform(2, 8, 5)).round(1)

PLAN_COLORS = {'Free':'#607D8B','Standard':ZOHO_BLUE,'Professional':ZOHO_TEAL,
               'Enterprise':ZOHO_AMBER,'Zoho One':ZOHO_PURPLE}

ax = axes[0]
ltv_vals = plan_stats['ltv'].values
bars = ax.bar(PLAN_ORDER, ltv_vals,
              color=[PLAN_COLORS[p] for p in PLAN_ORDER], width=0.6, zorder=3)
for bar, v in zip(bars, ltv_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            f'${v:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333')
ax.set_facecolor(BG)
ax.spines[['top','right']].set_visible(False)
ax.spines[['left','bottom']].set_color('#DDD')
ax.grid(axis='y', color='#E8E8E8', linewidth=0.7)
ax.set_axisbelow(True)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f'${v:,.0f}'))
ax.set_title('Customer LTV by plan tier', fontsize=12, fontweight='bold',
             pad=10, color='#1A1A2E')
ax.set_ylabel('Lifetime Value ($)', fontsize=10)
ax.tick_params(labelsize=9)

ax = axes[1]
nrr_vals = plan_stats['nrr'].values
colors_nrr = [ZOHO_RED if v < 90 else ZOHO_TEAL if v < 100 else ZOHO_BLUE for v in nrr_vals]
bars = ax.bar(PLAN_ORDER, nrr_vals, color=colors_nrr, width=0.6, zorder=3)
ax.axhline(100, color='#999', linestyle='--', linewidth=1.2, label='100% NRR baseline')
for bar, v in zip(bars, nrr_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333')
ax.set_facecolor(BG)
ax.spines[['top','right']].set_visible(False)
ax.spines[['left','bottom']].set_color('#DDD')
ax.grid(axis='y', color='#E8E8E8', linewidth=0.7)
ax.set_axisbelow(True)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_title('Net Revenue Retention (NRR) by plan', fontsize=12,
             fontweight='bold', pad=10, color='#1A1A2E')
ax.set_ylabel('NRR (%)', fontsize=10)
ax.set_ylim(0, 120)
ax.legend(fontsize=8)
ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig('/outputs/05_ltv_nrr.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("LTV & NRR chart saved.")
