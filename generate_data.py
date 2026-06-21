import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N = 5000
start_date = datetime(2022, 1, 1)

plan_tiers = {
    'Free':        {'price': 0,    'weight': 0.15},
    'Standard':    {'price': 20,   'weight': 0.30},
    'Professional':{'price': 35,   'weight': 0.25},
    'Enterprise':  {'price': 50,   'weight': 0.20},
    'Zoho One':    {'price': 105,  'weight': 0.10},
}
plan_names  = list(plan_tiers.keys())
plan_weights = [v['weight'] for v in plan_tiers.values()]
plan_prices  = {k: v['price'] for k, v in plan_tiers.items()}

industries = ['Technology', 'Retail', 'Healthcare', 'Finance', 'Education',
              'Manufacturing', 'Real Estate', 'Consulting']
countries  = ['India', 'United States', 'United Kingdom', 'Australia', 'Canada',
              'Germany', 'UAE', 'Singapore']
country_w  = [0.35, 0.30, 0.10, 0.06, 0.06, 0.05, 0.04, 0.04]

signup_days = np.random.randint(0, 365*3, N)
signup_dates = [start_date + timedelta(days=int(d)) for d in signup_days]

plans = np.random.choice(plan_names, size=N, p=plan_weights)

# Apps used: Zoho One uses many more apps
apps_map = {
    'Free': (1, 2), 'Standard': (1, 3), 'Professional': (2, 5),
    'Enterprise': (3, 8), 'Zoho One': (8, 20)
}
apps_used = [np.random.randint(*apps_map[p]) for p in plans]

# Monthly logins — more apps = more engagement
logins = []
for p, a in zip(plans, apps_used):
    base = {'Free':3,'Standard':8,'Professional':14,'Enterprise':20,'Zoho One':35}[p]
    logins.append(max(1, int(np.random.normal(base + a*0.5, base*0.4))))

# Support tickets — more complex plans raise more tickets
tickets = []
for p in plans:
    base = {'Free':0.2,'Standard':0.8,'Professional':1.5,'Enterprise':2.5,'Zoho One':3.0}[p]
    tickets.append(max(0, int(np.random.poisson(base))))

# Team size (seats)
seats = []
for p in plans:
    s = {'Free': (1,3), 'Standard': (1,5), 'Professional': (3,15),
         'Enterprise': (10,50), 'Zoho One': (5,100)}[p]
    seats.append(np.random.randint(*s))

company_size = np.random.choice(
    ['1-10','11-50','51-200','201-500','500+'], size=N,
    p=[0.30, 0.35, 0.20, 0.10, 0.05])

industries_col  = np.random.choice(industries, size=N)
countries_col   = np.random.choice(countries, size=N, p=country_w)

# Tenure in months
tenure = []
for sd in signup_dates:
    months = (datetime(2025, 6, 1) - sd).days // 30
    tenure.append(max(1, months))

# Churn probability — core business logic
# Low apps + low logins + Free/Standard plan + early tenure → higher churn
churn_flags = []
churn_months = []
for i in range(N):
    p   = plans[i]
    a   = apps_used[i]
    l   = logins[i]
    t   = tenure[i]
    tk  = tickets[i]

    base_churn = {
        'Free': 0.55, 'Standard': 0.35, 'Professional': 0.22,
        'Enterprise': 0.12, 'Zoho One': 0.06
    }[p]

    # Fewer apps → more likely to churn
    if a == 1: base_churn *= 1.5
    elif a >= 5: base_churn *= 0.5

    # Low logins → churn signal
    if l < 5:  base_churn *= 1.4
    elif l > 25: base_churn *= 0.6

    # Many unresolved tickets
    if tk > 3: base_churn *= 1.2

    # Danger zone: months 2-5
    if 2 <= t <= 5: base_churn *= 1.3

    base_churn = min(base_churn, 0.95)
    churned = np.random.random() < base_churn
    churn_flags.append(int(churned))

    if churned:
        # Churn happens in the first half of tenure
        cm = max(1, int(np.random.uniform(1, max(2, t * 0.6))))
    else:
        cm = None
    churn_months.append(cm)

# Upsell flag: currently on lower plan, high engagement → candidate for Zoho One
upsell_flags = []
for i in range(N):
    if plans[i] == 'Zoho One' or churn_flags[i] == 1:
        upsell_flags.append(0)
    else:
        score = 0
        if apps_used[i] >= 4: score += 2
        if logins[i] >= 15:   score += 2
        if seats[i] >= 10:    score += 1
        if tenure[i] >= 6:    score += 1
        upsell_flags.append(int(score >= 4))

df = pd.DataFrame({
    'customer_id':   [f'ZHO-{i:05d}' for i in range(N)],
    'signup_date':   signup_dates,
    'plan':          plans,
    'monthly_price': [plan_prices[p] for p in plans],
    'apps_used':     apps_used,
    'monthly_logins':logins,
    'support_tickets':tickets,
    'seats':         seats,
    'company_size':  company_size,
    'industry':      industries_col,
    'country':       countries_col,
    'tenure_months': tenure,
    'churned':       churn_flags,
    'churn_month':   churn_months,
    'upsell_flag':   upsell_flags,
})

df.to_csv('/outputs/zoho_smb_data.csv', index=False)
print(f"Dataset created: {len(df)} customers")
print(df['plan'].value_counts())
print(f"\nOverall churn rate: {df['churned'].mean():.1%}")
print(f"Upsell candidates: {df['upsell_flag'].sum()}")
