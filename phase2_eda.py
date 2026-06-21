import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('zoho_smb_data.csv', parse_dates=['signup_date'])

ZOHO_RED    = '#E42527'
ZOHO_BLUE   = '#1565C0'
ZOHO_TEAL   = '#00897B'
ZOHO_AMBER  = '#F57C00'
ZOHO_PURPLE = '#5E35B1'
GRAY        = '#607D8B'
BG          = '#F8F9FA'
PLAN_COLORS = {
    'Free': GRAY, 'Standard': ZOHO_BLUE, 'Professional': ZOHO_TEAL,
    'Enterprise': ZOHO_AMBER, 'Zoho One': ZOHO_PURPLE
}
PLAN_ORDER = ['Free', 'Standard', 'Professional', 'Enterprise', 'Zoho One']

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(BG)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#DDD')
    ax.grid(axis='y', color='#E8E8E8', linewidth=0.7)
    ax.set_axisbelow(True)
    if title:  ax.set_title(title, fontsize=12, fontweight='bold', pad=10, color='#1A1A2E')
    if xlabel: ax.set_xlabel(xlabel, fontsize=10, color='#555')
    if ylabel: ax.set_ylabel(ylabel, fontsize=10, color='#555')
    ax.tick_params(colors='#555', labelsize=9)

# ─── FIGURE 1: EDA Overview (2×2) ───────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('white')
fig.suptitle('Zoho SMB — Customer Churn EDA', fontsize=16, fontweight='bold',
             color='#1A1A2E', y=1.01)

# 1A: Churn rate by plan
ax = axes[0, 0]
churn_by_plan = df.groupby('plan')['churned'].mean().reindex(PLAN_ORDER) * 100
bars = ax.bar(PLAN_ORDER, churn_by_plan,
              color=[PLAN_COLORS[p] for p in PLAN_ORDER], width=0.6, zorder=3)
for bar, v in zip(bars, churn_by_plan):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
style_ax(ax, 'Churn rate by plan tier', 'Plan', 'Churn rate (%)')
ax.set_ylim(0, churn_by_plan.max() * 1.2)

# 1B: Churn rate by apps used (binned)
ax = axes[0, 1]
df['apps_bin'] = pd.cut(df['apps_used'], bins=[0,1,2,4,8,30],
                        labels=['1 app','2 apps','3–4 apps','5–8 apps','9+ apps'])
churn_apps = df.groupby('apps_bin', observed=True)['churned'].mean() * 100
colors_apps = [ZOHO_RED, '#E57373', ZOHO_AMBER, ZOHO_TEAL, ZOHO_PURPLE]
bars = ax.bar(churn_apps.index, churn_apps, color=colors_apps, width=0.6, zorder=3)
for bar, v in zip(bars, churn_apps):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
style_ax(ax, 'Churn rate by number of apps used', 'Apps used', 'Churn rate (%)')
ax.set_ylim(0, churn_apps.max() * 1.2)

# 1C: Churn rate by tenure bucket
ax = axes[1, 0]
df['tenure_bin'] = pd.cut(df['tenure_months'], bins=[0,3,6,12,24,100],
                          labels=['0–3 mo','3–6 mo','6–12 mo','12–24 mo','24+ mo'])
churn_tenure = df.groupby('tenure_bin', observed=True)['churned'].mean() * 100
danger_color = [ZOHO_RED if i < 2 else ZOHO_BLUE for i in range(len(churn_tenure))]
bars = ax.bar(churn_tenure.index, churn_tenure, color=danger_color, width=0.6, zorder=3)
for bar, v in zip(bars, churn_tenure):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
style_ax(ax, 'Churn rate by customer tenure', 'Tenure', 'Churn rate (%)')
ax.set_ylim(0, churn_tenure.max() * 1.3)
red_patch  = mpatches.Patch(color=ZOHO_RED,  label='Danger zone (high churn)')
blue_patch = mpatches.Patch(color=ZOHO_BLUE, label='Stable zone')
ax.legend(handles=[red_patch, blue_patch], fontsize=8, framealpha=0.7)

# 1D: Customer distribution by plan
ax = axes[1, 1]
dist = df['plan'].value_counts().reindex(PLAN_ORDER)
wedges, texts, autotexts = ax.pie(
    dist, labels=PLAN_ORDER, autopct='%1.1f%%',
    colors=[PLAN_COLORS[p] for p in PLAN_ORDER],
    startangle=140, pctdistance=0.75,
    wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'})
for t in autotexts: t.set_fontsize(9); t.set_fontweight('bold'); t.set_color('white')
for t in texts: t.set_fontsize(10); t.set_color('#333')
ax.set_title('Customer distribution by plan', fontsize=12,
             fontweight='bold', pad=10, color='#1A1A2E')

plt.tight_layout()
plt.savefig('/home/claude/zoho_churn_project/outputs/01_eda_overview.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("EDA chart saved.")

# ─── FIGURE 2: Country & Industry breakdown ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('white')

ax = axes[0]
top_countries = df.groupby('country')['churned'].agg(['mean','count']).sort_values('mean', ascending=True)
top_countries['mean'] *= 100
colors_c = [ZOHO_RED if v > 35 else ZOHO_BLUE for v in top_countries['mean']]
bars = ax.barh(top_countries.index, top_countries['mean'], color=colors_c, height=0.6, zorder=3)
for bar, v in zip(bars, top_countries['mean']):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{v:.1f}%', va='center', fontsize=9, fontweight='bold', color='#333')
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
style_ax(ax, 'Churn rate by country', 'Churn rate (%)', '')
ax.set_xlim(0, top_countries['mean'].max() * 1.2)
ax.axvline(x=df['churned'].mean()*100, color='#999', linestyle='--', linewidth=1, label='Avg churn')
ax.legend(fontsize=8)

ax = axes[1]
top_ind = df.groupby('industry')['churned'].mean().sort_values(ascending=True) * 100
colors_i = [ZOHO_RED if v > 38 else ZOHO_TEAL for v in top_ind]
bars = ax.barh(top_ind.index, top_ind, color=colors_i, height=0.6, zorder=3)
for bar, v in zip(bars, top_ind):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{v:.1f}%', va='center', fontsize=9, fontweight='bold', color='#333')
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
style_ax(ax, 'Churn rate by industry', 'Churn rate (%)', '')
ax.set_xlim(0, top_ind.max() * 1.2)

plt.tight_layout()
plt.savefig('/outputs/02_eda_segmentation.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Segmentation chart saved.")
