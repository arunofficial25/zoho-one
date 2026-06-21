import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('zoho_smb_data.csv', parse_dates=['signup_date'])

ZOHO_RED    = '#E42527'
ZOHO_BLUE   = '#1565C0'
ZOHO_TEAL   = '#00897B'
ZOHO_AMBER  = '#F57C00'
ZOHO_PURPLE = '#5E35B1'
BG = '#F8F9FA'

# ─── Feature Engineering ────────────────────────────────────────────────────
PLAN_PRICES = {'Free':0,'Standard':20,'Professional':35,'Enterprise':50,'Zoho One':105}
le_plan = LabelEncoder()
le_ind  = LabelEncoder()
le_cty  = LabelEncoder()
le_sz   = LabelEncoder()

df['plan_encoded']    = le_plan.fit_transform(df['plan'])
df['industry_enc']    = le_ind.fit_transform(df['industry'])
df['country_enc']     = le_cty.fit_transform(df['country'])
df['company_size_enc']= le_sz.fit_transform(df['company_size'])
df['monthly_price']   = df['plan'].map(PLAN_PRICES)
df['revenue_per_seat']= df['monthly_price'] / df['seats'].clip(1)
df['login_per_app']   = df['monthly_logins'] / df['apps_used'].clip(1)
df['engagement_score']= (df['monthly_logins'] * 0.4 + df['apps_used'] * 2 + df['seats'] * 0.3)

FEATURES = ['plan_encoded','monthly_price','apps_used','monthly_logins',
            'support_tickets','seats','tenure_months','revenue_per_seat',
            'login_per_app','engagement_score','industry_enc','country_enc',
            'company_size_enc']

# ─── CHURN MODEL ────────────────────────────────────────────────────────────
X = df[FEATURES]
y_churn  = df['churned']
y_upsell = df['upsell_flag']

X_train, X_test, y_train, y_test = train_test_split(X, y_churn, test_size=0.2,
                                                     random_state=42, stratify=y_churn)

churn_model = RandomForestClassifier(n_estimators=200, max_depth=8,
                                     min_samples_leaf=10, random_state=42)
churn_model.fit(X_train, y_train)
y_pred_proba = churn_model.predict_proba(X_test)[:,1]
y_pred       = churn_model.predict(X_test)
auc_score    = roc_auc_score(y_test, y_pred_proba)
cv_scores    = cross_val_score(churn_model, X, y_churn, cv=5, scoring='roc_auc')

print(f"Churn Model AUC: {auc_score:.3f}")
print(f"Cross-val AUC:   {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(classification_report(y_test, y_pred))

# ─── UPSELL MODEL ───────────────────────────────────────────────────────────
# only non-churned, non-ZohoOne customers
upsell_df = df[df['churned'] == 0][df['plan'] != 'Zoho One']
X_up = upsell_df[FEATURES]
y_up = upsell_df['upsell_flag']

Xu_train, Xu_test, yu_train, yu_test = train_test_split(X_up, y_up, test_size=0.2,
                                                         random_state=42, stratify=y_up)
upsell_model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
upsell_model.fit(Xu_train, yu_train)
yu_pred_proba = upsell_model.predict_proba(Xu_test)[:,1]
up_auc = roc_auc_score(yu_test, yu_pred_proba)
print(f"\nUpsell Model AUC: {up_auc:.3f}")

# Attach scores to full dataset
df['churn_score']  = (churn_model.predict_proba(df[FEATURES])[:,1] * 100).round(1)
df['upsell_score'] = 0.0
mask = (df['churned'] == 0) & (df['plan'] != 'Zoho One')
df.loc[mask, 'upsell_score'] = (upsell_model.predict_proba(df.loc[mask, FEATURES])[:,1] * 100).round(1)
df.to_csv('/home/claude/zoho_churn_project/zoho_scored_data.csv', index=False)

# ─── FIGURE 6: Model performance & feature importance ───────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('white')
fig.suptitle('Zoho Churn & Upsell Prediction Models', fontsize=16,
             fontweight='bold', color='#1A1A2E', y=1.01)

# 6A: ROC Curve
ax = axes[0, 0]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
ax.plot(fpr, tpr, color=ZOHO_BLUE, lw=2.5, label=f'Churn model (AUC = {auc_score:.3f})')
fpr_u, tpr_u, _ = roc_curve(yu_test, yu_pred_proba)
ax.plot(fpr_u, tpr_u, color=ZOHO_TEAL, lw=2.5, label=f'Upsell model (AUC = {up_auc:.3f})')
ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.4, label='Random (AUC=0.5)')
ax.set_facecolor(BG)
ax.spines[['top','right']].set_visible(False)
ax.set_xlabel('False Positive Rate', fontsize=10)
ax.set_ylabel('True Positive Rate', fontsize=10)
ax.set_title('ROC curves — churn & upsell models', fontsize=12, fontweight='bold', color='#1A1A2E')
ax.legend(fontsize=9)
ax.fill_between(fpr, tpr, alpha=0.08, color=ZOHO_BLUE)

# 6B: Feature importance (churn model)
ax = axes[0, 1]
feat_imp = pd.Series(churn_model.feature_importances_, index=FEATURES).sort_values(ascending=True)
feat_labels = {
    'plan_encoded':'Plan tier','monthly_price':'Monthly price',
    'apps_used':'Apps used','monthly_logins':'Monthly logins',
    'support_tickets':'Support tickets','seats':'Seats',
    'tenure_months':'Tenure (months)','revenue_per_seat':'Revenue/seat',
    'login_per_app':'Logins/app','engagement_score':'Engagement score',
    'industry_enc':'Industry','country_enc':'Country','company_size_enc':'Company size'
}
feat_imp.index = [feat_labels.get(f, f) for f in feat_imp.index]
colors_fi = [ZOHO_RED if v > 0.12 else ZOHO_BLUE if v > 0.06 else '#B0BEC5'
             for v in feat_imp.values]
ax.barh(feat_imp.index, feat_imp.values, color=colors_fi, height=0.7, zorder=3)
ax.set_facecolor(BG)
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='x', color='#E8E8E8', linewidth=0.7)
ax.set_axisbelow(True)
ax.set_title('Feature importance — churn model', fontsize=12, fontweight='bold', color='#1A1A2E')
ax.set_xlabel('Importance score', fontsize=10)
ax.tick_params(labelsize=9)

# 6C: Churn risk distribution
ax = axes[1, 0]
churned_scores     = df[df['churned'] == 1]['churn_score']
not_churned_scores = df[df['churned'] == 0]['churn_score']
ax.hist(not_churned_scores, bins=30, alpha=0.6, color=ZOHO_BLUE, label='Retained', density=True)
ax.hist(churned_scores,     bins=30, alpha=0.6, color=ZOHO_RED,  label='Churned',  density=True)
ax.set_facecolor(BG)
ax.spines[['top','right']].set_visible(False)
ax.set_xlabel('Churn probability score (0–100)', fontsize=10)
ax.set_ylabel('Density', fontsize=10)
ax.set_title('Churn score distribution', fontsize=12, fontweight='bold', color='#1A1A2E')
ax.legend(fontsize=9)
ax.axvline(50, color='#999', linestyle='--', linewidth=1)
ax.text(52, ax.get_ylim()[1]*0.9, 'Threshold', fontsize=8, color='#777')

# 6D: At-risk customer matrix (churn score vs upsell score)
ax = axes[1, 1]
sample = df[(df['plan'] != 'Zoho One') & (df['churned']==0)].sample(500, random_state=42)
colors_q = []
for _, row in sample.iterrows():
    if row['churn_score'] > 60 and row['upsell_score'] > 50:
        colors_q.append(ZOHO_AMBER)   # high risk + upsell opportunity
    elif row['churn_score'] > 60:
        colors_q.append(ZOHO_RED)     # high churn risk
    elif row['upsell_score'] > 50:
        colors_q.append(ZOHO_TEAL)    # upsell ready
    else:
        colors_q.append('#90CAF9')    # healthy

ax.scatter(sample['churn_score'], sample['upsell_score'],
           c=colors_q, alpha=0.6, s=25, zorder=3)
ax.axvline(60, color='#999', linestyle='--', linewidth=1)
ax.axhline(50, color='#999', linestyle='--', linewidth=1)
ax.set_facecolor(BG)
ax.spines[['top','right']].set_visible(False)
ax.set_xlabel('Churn risk score', fontsize=10)
ax.set_ylabel('Upsell readiness score', fontsize=10)
ax.set_title('Churn risk vs upsell opportunity matrix', fontsize=12, fontweight='bold', color='#1A1A2E')

from matplotlib.lines import Line2D
legend_els = [
    Line2D([0],[0],marker='o',color='w',markerfacecolor=ZOHO_RED,   ms=8, label='High churn risk'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=ZOHO_TEAL,  ms=8, label='Upsell ready'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=ZOHO_AMBER, ms=8, label='Urgent: risk + upsell'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor='#90CAF9',  ms=8, label='Healthy'),
]
ax.legend(handles=legend_els, fontsize=8, framealpha=0.8)
ax.text(62, 52, 'Urgent\naction', fontsize=8, color=ZOHO_AMBER, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/06_ml_model.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("ML model chart saved.")

# ─── Business Impact Summary ────────────────────────────────────────────────
at_risk      = df[(df['churn_score'] > 60) & (df['churned']==0)]
upsell_ready = df[(df['upsell_score'] > 60) & (df['churned']==0) & (df['plan'] != 'Zoho One')]
monthly_at_risk_mrr = (at_risk['monthly_price'] * at_risk['seats']).sum()
upsell_potential    = upsell_ready.shape[0] * (105 - upsell_ready['monthly_price'].mean()) * upsell_ready['seats'].mean()

print(f"\n{'='*55}")
print(f"  BUSINESS IMPACT SUMMARY")
print(f"{'='*55}")
print(f"  Customers at high churn risk (score > 60): {len(at_risk):,}")
print(f"  MRR at risk:                              ${monthly_at_risk_mrr:,.0f}/mo")
print(f"  Upsell-ready customers (score > 60):      {len(upsell_ready):,}")
print(f"  Upsell revenue potential:                 ${upsell_potential:,.0f}/mo")
print(f"  Net opportunity (save churn + upsell):    ${monthly_at_risk_mrr + upsell_potential:,.0f}/mo")
print(f"{'='*55}")
