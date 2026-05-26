"""
PROJECT 1: BANKING CUSTOMER CHURN ANALYSIS
============================================
Objective: Analyze customer churn patterns, identify at-risk segments, 
and recommend retention strategies.

Skills demonstrated: Python, Data Cleaning, EDA, Statistical Analysis, Visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# SECTION 1: DATA LOADING & UNDERSTANDING
# ============================================================================

print("=" * 70)
print("BANKING CUSTOMER CHURN ANALYSIS")
print("=" * 70)

# Create sample banking churn dataset
np.random.seed(42)
n_customers = 10000

data = {
    'CustomerID': range(1001, 1001 + n_customers),
    'Age': np.random.normal(45, 15, n_customers).astype(int).clip(18, 80),
    'Tenure_Years': np.random.randint(0, 11, n_customers),
    'Balance': np.random.uniform(0, 250000, n_customers),
    'CreditScore': np.random.normal(700, 80, n_customers).astype(int).clip(300, 850),
    'NumOfProducts': np.random.randint(1, 5, n_customers),
    'IsActiveMember': np.random.choice([0, 1], n_customers, p=[0.4, 0.6]),
    'EstimatedSalary': np.random.uniform(10000, 200000, n_customers),
    'MonthlyTransactions': np.random.randint(5, 50, n_customers),
    'ComplaintsFiled': np.random.randint(0, 3, n_customers),
}

# Create churn column (with realistic patterns)
churn_probability = []
for i in range(n_customers):
    prob = 0.2  # base probability
    
    # Factors increasing churn risk
    if data['Tenure_Years'][i] < 2:
        prob += 0.15  # New customers are riskier
    if data['Balance'][i] < 10000:
        prob += 0.1  # Low balance = higher risk
    if data['IsActiveMember'][i] == 0:
        prob += 0.25  # Inactive = high risk
    if data['ComplaintsFiled'][i] > 0:
        prob += 0.15 * data['ComplaintsFiled'][i]  # Complaints increase risk
    if data['NumOfProducts'][i] == 1:
        prob += 0.05  # Single product customers slightly higher risk
    
    prob = min(prob, 0.95)  # Cap probability at 95%
    churn_probability.append(prob)

data['Churn'] = [1 if np.random.random() < p else 0 for p in churn_probability]

df = pd.DataFrame(data)

print("\n✓ Dataset Created Successfully")
print(f"  Total Records: {len(df):,}")
print(f"  Total Columns: {df.shape[1]}")
print(f"\nFirst few records:")
print(df.head())

# ============================================================================
# SECTION 2: DATA QUALITY ASSESSMENT
# ============================================================================

print("\n" + "=" * 70)
print("DATA QUALITY ASSESSMENT")
print("=" * 70)

print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nBasic Statistics:\n{df.describe().round(2)}")

# ============================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 70)

# Overall Churn Rate
churn_rate = (df['Churn'].sum() / len(df)) * 100
print(f"\n📊 Overall Churn Rate: {churn_rate:.2f}%")
print(f"   Churned Customers: {df['Churn'].sum():,}")
print(f"   Retained Customers: {(df['Churn'] == 0).sum():,}")

# Churn by Age Group
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100], 
                        labels=['<30', '30-40', '40-50', '50-60', '60+'])
print(f"\n📈 Churn Rate by Age Group:")
age_churn = df.groupby('AgeGroup', observed=True)['Churn'].agg(['count', 'sum', lambda x: (x.sum()/len(x))*100])
age_churn.columns = ['Total', 'Churned', 'ChurnRate%']
print(age_churn.round(2))

# Churn by Tenure
print(f"\n📈 Churn Rate by Tenure:")
tenure_churn = df.groupby('Tenure_Years')['Churn'].agg(['count', 'sum', lambda x: (x.sum()/len(x))*100])
tenure_churn.columns = ['Total', 'Churned', 'ChurnRate%']
print(tenure_churn.round(2))

# Account Activity Impact
print(f"\n📈 Impact of Account Activity on Churn:")
activity_churn = df.groupby('IsActiveMember')['Churn'].agg(['count', 'sum', lambda x: (x.sum()/len(x))*100])
activity_churn.columns = ['Total', 'Churned', 'ChurnRate%']
activity_churn.index = ['Inactive', 'Active']
print(activity_churn.round(2))

# ============================================================================
# SECTION 4: RISK SEGMENTATION
# ============================================================================

print("\n" + "=" * 70)
print("CUSTOMER RISK SEGMENTATION")
print("=" * 70)

# Create risk score based on multiple factors
df['RiskScore'] = 0

# Tenure (newer customers = higher risk)
df.loc[df['Tenure_Years'] < 1, 'RiskScore'] += 30
df.loc[(df['Tenure_Years'] >= 1) & (df['Tenure_Years'] < 3), 'RiskScore'] += 20
df.loc[(df['Tenure_Years'] >= 3) & (df['Tenure_Years'] < 5), 'RiskScore'] += 10

# Activity Status (inactive = high risk)
df.loc[df['IsActiveMember'] == 0, 'RiskScore'] += 25

# Account Balance (low balance = higher risk)
balance_percentile = df['Balance'].quantile([0.33, 0.67])
df.loc[df['Balance'] < balance_percentile[0.33], 'RiskScore'] += 15
df.loc[df['Balance'] < balance_percentile[0.67], 'RiskScore'] += 5

# Complaints (complaints = high risk)
df.loc[df['ComplaintsFiled'] > 0, 'RiskScore'] += (10 * df['ComplaintsFiled'])

# Product diversity (single product = slightly higher risk)
df.loc[df['NumOfProducts'] == 1, 'RiskScore'] += 5

# Monthly transactions (low activity = risk)
transaction_threshold = df['MonthlyTransactions'].quantile(0.33)
df.loc[df['MonthlyTransactions'] < transaction_threshold, 'RiskScore'] += 10

# Cap risk score at 100
df['RiskScore'] = df['RiskScore'].clip(0, 100)

# Categorize risk levels
def categorize_risk(score):
    if score >= 70:
        return 'High Risk'
    elif score >= 40:
        return 'Medium Risk'
    else:
        return 'Low Risk'

df['RiskCategory'] = df['RiskScore'].apply(categorize_risk)

print(f"\n🎯 Risk Segment Distribution:")
risk_distribution = df['RiskCategory'].value_counts()
risk_pct = (df['RiskCategory'].value_counts(normalize=True) * 100).round(2)

for category in ['High Risk', 'Medium Risk', 'Low Risk']:
    if category in risk_distribution.index:
        count = risk_distribution[category]
        pct = risk_pct[category]
        print(f"   {category}: {count:,} customers ({pct}%)")

print(f"\n📊 Churn Rate by Risk Segment:")
risk_churn = df.groupby('RiskCategory', observed=True)['Churn'].agg(
    ['count', 'sum', lambda x: (x.sum()/len(x))*100]
)
risk_churn.columns = ['Total', 'Churned', 'ChurnRate%']
# Reorder by risk level
risk_churn = risk_churn.reindex(['High Risk', 'Medium Risk', 'Low Risk'])
print(risk_churn.round(2))

# ============================================================================
# SECTION 5: KEY FINDINGS & RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 70)
print("KEY FINDINGS & RETENTION RECOMMENDATIONS")
print("=" * 70)

# High-risk customer analysis
high_risk = df[df['RiskCategory'] == 'High Risk']
print(f"\n⚠️  HIGH-RISK SEGMENT PROFILE:")
print(f"   • Average Tenure: {high_risk['Tenure_Years'].mean():.1f} years")
print(f"   • Inactive Rate: {(high_risk['IsActiveMember'] == 0).sum() / len(high_risk) * 100:.1f}%")
print(f"   • Average Balance: ${high_risk['Balance'].mean():,.0f}")
print(f"   • Actual Churn Rate: {(high_risk['Churn'].sum() / len(high_risk) * 100):.1f}%")

print(f"\n💡 RECOMMENDATIONS:")
print(f"   1. Implement proactive outreach for inactive high-risk customers")
print(f"   2. Offer loyalty rewards for customers < 2 years tenure")
print(f"   3. Create complaint resolution task force (FY + complaints)")
print(f"   4. Develop cross-sell strategy for single-product customers")
print(f"   5. Monthly engagement initiatives for accounts with low transaction volume")

# ============================================================================
# SECTION 6: VISUALIZATIONS
# ============================================================================

print(f"\n📊 Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Churn Rate by Tenure
ax1 = axes[0, 0]
tenure_data = df.groupby('Tenure_Years')['Churn'].apply(lambda x: (x.sum()/len(x))*100)
ax1.bar(tenure_data.index, tenure_data.values, color='#E74C3C', alpha=0.7)
ax1.set_xlabel('Tenure (Years)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Churn Rate (%)', fontsize=11, fontweight='bold')
ax1.set_title('Churn Rate by Customer Tenure', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Risk Segment Distribution
ax2 = axes[0, 1]
risk_counts = df['RiskCategory'].value_counts()
risk_order = ['High Risk', 'Medium Risk', 'Low Risk']
risk_counts = risk_counts.reindex(risk_order)
colors = ['#E74C3C', '#F39C12', '#27AE60']
ax2.bar(risk_counts.index, risk_counts.values, color=colors, alpha=0.7)
ax2.set_ylabel('Number of Customers', fontsize=11, fontweight='bold')
ax2.set_title('Risk Segment Distribution', fontsize=12, fontweight='bold')
for i, v in enumerate(risk_counts.values):
    ax2.text(i, v + 100, str(v), ha='center', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Activity Status Impact
ax3 = axes[1, 0]
activity_data = df.groupby('IsActiveMember')['Churn'].apply(lambda x: (x.sum()/len(x))*100)
activity_labels = ['Inactive', 'Active']
ax3.bar(activity_labels, activity_data.values, color=['#E74C3C', '#27AE60'], alpha=0.7)
ax3.set_ylabel('Churn Rate (%)', fontsize=11, fontweight='bold')
ax3.set_title('Impact of Account Activity on Churn', fontsize=12, fontweight='bold')
for i, v in enumerate(activity_data.values):
    ax3.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Risk Score Distribution by Churn Status
ax4 = axes[1, 1]
churned = df[df['Churn'] == 1]['RiskScore']
retained = df[df['Churn'] == 0]['RiskScore']
ax4.hist(retained, bins=30, alpha=0.6, label='Retained', color='#27AE60')
ax4.hist(churned, bins=30, alpha=0.6, label='Churned', color='#E74C3C')
ax4.set_xlabel('Risk Score', fontsize=11, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax4.set_title('Risk Score Distribution by Churn Status', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('churn_analysis_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Dashboard saved as 'churn_analysis_dashboard.png'")

# ============================================================================
# SECTION 7: EXPORT RESULTS
# ============================================================================

# Export high-risk customer list for targeted retention
high_risk_export = df[df['RiskCategory'] == 'High Risk'][
    ['CustomerID', 'Age', 'Tenure_Years', 'Balance', 'IsActiveMember', 
     'RiskScore', 'RiskCategory', 'Churn']
].sort_values('RiskScore', ascending=False)

high_risk_export.to_csv('high_risk_customers.csv', index=False)
print(f"✓ High-risk customer list exported: 'high_risk_customers.csv' ({len(high_risk_export)} records)")

# Export summary statistics
summary_stats = {
    'Total Customers': len(df),
    'Churned Customers': df['Churn'].sum(),
    'Overall Churn Rate (%)': churn_rate,
    'High Risk Customers': len(df[df['RiskCategory'] == 'High Risk']),
    'High Risk Churn Rate (%)': (df[df['RiskCategory'] == 'High Risk']['Churn'].sum() / len(df[df['RiskCategory'] == 'High Risk']) * 100),
}

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
for key, value in summary_stats.items():
    if '(%)' in key:
        print(f"{key}: {value:.2f}%")
    else:
        print(f"{key}: {value:,}")
