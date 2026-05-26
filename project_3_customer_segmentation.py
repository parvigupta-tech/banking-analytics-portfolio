"""
PROJECT 3: CUSTOMER SEGMENTATION & CAMPAIGN ANALYTICS
===================================================
Objective: Segment customers using RFM analysis, analyze campaign effectiveness,
and provide targeted marketing recommendations.

Skills demonstrated: Python, RFM Analysis, Customer Analytics, A/B Testing Insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 9)

print("=" * 80)
print("CUSTOMER SEGMENTATION & CAMPAIGN ANALYTICS")
print("=" * 80)

# ============================================================================
# SECTION 1: GENERATE CUSTOMER TRANSACTION DATA
# ============================================================================

print("\n📁 Generating Customer Transaction Data...")

np.random.seed(42)
n_customers = 8000

# Create customer base
customer_data = {
    'CustomerID': [f'CUST{i+100000}' for i in range(n_customers)],
    'JoinDate': [
        (datetime.now() - timedelta(days=np.random.randint(30, 1095))).strftime('%Y-%m-%d')
        for _ in range(n_customers)
    ],
    'Age': np.random.normal(45, 15, n_customers).astype(int).clip(18, 80),
    'Gender': np.random.choice(['M', 'F'], n_customers),
    'ProductType': np.random.choice(['Savings', 'Checking', 'Credit Card', 'Loan', 'Investment'], n_customers),
}

# Generate transactions
transactions = []
for i in range(n_customers):
    customer_id = customer_data['CustomerID'][i]
    join_date = datetime.strptime(customer_data['JoinDate'][i], '%Y-%m-%d')
    
    # Number of transactions varies by customer
    n_transactions = np.random.choice(
        [0, 1, 2, 3, 4, 5, 6, 8, 10, 15],
        p=[0.05, 0.1, 0.1, 0.15, 0.15, 0.15, 0.12, 0.1, 0.06, 0.02]
    )
    
    for j in range(n_transactions):
        days_since_join = (datetime.now() - join_date).days
        if days_since_join <= 0:
            days_since_join = 1
        
        transaction_date = join_date + timedelta(
            days=np.random.randint(0, min(days_since_join, 365))
        )
        
        # Transaction amount varies by product type
        product = customer_data['ProductType'][i]
        if product == 'Checking':
            amount = np.random.uniform(100, 5000)
        elif product == 'Savings':
            amount = np.random.uniform(1000, 20000)
        elif product == 'Credit Card':
            amount = np.random.uniform(50, 3000)
        elif product == 'Loan':
            amount = np.random.uniform(5000, 100000)
        else:  # Investment
            amount = np.random.uniform(2000, 50000)
        
        transactions.append({
            'CustomerID': customer_id,
            'TransactionDate': transaction_date.strftime('%Y-%m-%d'),
            'Amount': amount,
            'ProductType': product
        })

transactions_df = pd.DataFrame(transactions)

# Create customer dataframe
customers_df = pd.DataFrame(customer_data)
customers_df['JoinDate'] = pd.to_datetime(customers_df['JoinDate'])

print(f"✓ Generated data for {len(customers_df):,} customers and {len(transactions_df):,} transactions")

# ============================================================================
# SECTION 2: RFM ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("RFM ANALYSIS (Recency, Frequency, Monetary)")
print("=" * 80)

# Convert transaction date
transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])

# Reference date (today)
reference_date = datetime.now()

# Calculate RFM metrics
rfm = transactions_df.groupby('CustomerID').agg({
    'TransactionDate': lambda x: (reference_date - x.max()).days,  # Recency
    'CustomerID': 'count',  # Frequency
    'Amount': 'sum'  # Monetary
}).rename(columns={
    'TransactionDate': 'Recency',
    'CustomerID': 'Frequency',
    'Amount': 'Monetary'
})

# Add customers with no transactions
no_transaction_customers = customers_df[~customers_df['CustomerID'].isin(rfm.index)]
for cust_id in no_transaction_customers['CustomerID']:
    rfm.loc[cust_id] = [999, 0, 0]  # 999 days = never transacted

rfm = rfm.reset_index()
rfm['Monetary'] = rfm['Monetary'].fillna(0)
rfm['Frequency'] = rfm['Frequency'].fillna(0)

print(f"\n📊 RFM STATISTICS:")
print(rfm[['Recency', 'Frequency', 'Monetary']].describe().round(2))

# ============================================================================
# SECTION 3: RFM SCORING & SEGMENTATION
# ============================================================================

print("\n" + "=" * 80)
print("CUSTOMER SEGMENTATION")
print("=" * 80)

# Create RFM scores (quartile-based, 1-4 where 4 is best)
rfm['R_Score'] = pd.qcut(rfm['Recency'], q=4, labels=[4, 3, 2, 1], duplicates='drop')
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=4, labels=[1, 2, 3, 4], duplicates='drop')
rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), q=4, labels=[1, 2, 3, 4], duplicates='drop')

# Convert to numeric
rfm['R_Score'] = rfm['R_Score'].astype(int)
rfm['F_Score'] = rfm['F_Score'].astype(int)
rfm['M_Score'] = rfm['M_Score'].astype(int)

# Overall RFM Score
rfm['RFM_Score'] = rfm['R_Score'] * 100 + rfm['F_Score'] * 10 + rfm['M_Score']

# Customer Segmentation Logic
def segment_customer(row):
    r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
    
    # Champions: Best recent, frequent, and high value
    if r >= 3 and f >= 3 and m >= 3:
        return 'Champions'
    # Loyal Customers: Good frequency and monetary value
    elif f >= 3 and m >= 3:
        return 'Loyal Customers'
    # Potential Loyalists: Recent, good frequency
    elif r >= 3 and f >= 2:
        return 'Potential Loyalists'
    # At Risk: High value but not recent
    elif m >= 3 and r <= 2:
        return 'At Risk'
    # Need Attention: Recent but low frequency/value
    elif r >= 3 and f <= 2 and m <= 2:
        return 'Need Attention'
    # About to Sleep: Low recency, some frequency
    elif r <= 2 and f >= 2:
        return 'About to Sleep'
    # Lost Customers: No recent activity
    elif r <= 1 and f <= 1:
        return 'Lost'
    else:
        return 'Other'

rfm['Segment'] = rfm.apply(segment_customer, axis=1)

# Segment Distribution
print(f"\n🎯 CUSTOMER SEGMENTATION DISTRIBUTION:")
segment_dist = rfm['Segment'].value_counts().sort_values(ascending=False)
print(segment_dist)

# Segment Analysis
print(f"\n📊 DETAILED SEGMENT ANALYSIS:")
segment_analysis = rfm.groupby('Segment').agg({
    'CustomerID': 'count',
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'sum']
}).round(2)

segment_analysis.columns = ['Count', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Total_Revenue']
print(segment_analysis)

# ============================================================================
# SECTION 4: CAMPAIGN PERFORMANCE ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("CAMPAIGN ANALYTICS")
print("=" * 80)

# Create campaign data
np.random.seed(42)
campaigns = []

campaign_segments = {
    'Champions': {'response_rate': 0.45, 'conversion_rate': 0.35, 'email_cost': 0.5},
    'Loyal Customers': {'response_rate': 0.35, 'conversion_rate': 0.25, 'email_cost': 0.5},
    'Potential Loyalists': {'response_rate': 0.25, 'conversion_rate': 0.15, 'email_cost': 0.5},
    'Need Attention': {'response_rate': 0.20, 'conversion_rate': 0.10, 'email_cost': 0.5},
    'At Risk': {'response_rate': 0.15, 'conversion_rate': 0.08, 'email_cost': 0.5},
    'About to Sleep': {'response_rate': 0.10, 'conversion_rate': 0.05, 'email_cost': 0.5},
    'Lost': {'response_rate': 0.05, 'conversion_rate': 0.02, 'email_cost': 0.5},
}

for idx, row in rfm.iterrows():
    segment = row['Segment']
    metrics = campaign_segments.get(segment, {'response_rate': 0.10, 'conversion_rate': 0.05, 'email_cost': 0.5})
    
    # Simulate campaign response
    responded = np.random.random() < metrics['response_rate']
    converted = responded and (np.random.random() < metrics['conversion_rate'])
    
    revenue_if_converted = np.random.uniform(500, 5000) if converted else 0
    
    campaigns.append({
        'CustomerID': row['CustomerID'],
        'Segment': segment,
        'EmailSent': 1,
        'EmailOpened': 1 if (np.random.random() < metrics['response_rate'] * 1.2) else 0,
        'EmailClicked': 1 if responded else 0,
        'Converted': 1 if converted else 0,
        'RevenueGenerated': revenue_if_converted,
        'CostPerEmail': metrics['email_cost']
    })

campaign_df = pd.DataFrame(campaigns)

# Campaign Performance Metrics
print(f"\n📈 CAMPAIGN PERFORMANCE METRICS:")

campaign_performance = campaign_df.groupby('Segment').agg({
    'EmailSent': 'sum',
    'EmailOpened': 'sum',
    'EmailClicked': 'sum',
    'Converted': 'sum',
    'RevenueGenerated': 'sum',
    'CostPerEmail': lambda x: x.iloc[0]
}).round(2)

campaign_performance['OpenRate%'] = (campaign_performance['EmailOpened'] / campaign_performance['EmailSent'] * 100).round(2)
campaign_performance['ClickRate%'] = (campaign_performance['EmailClicked'] / campaign_performance['EmailSent'] * 100).round(2)
campaign_performance['ConversionRate%'] = (campaign_performance['Converted'] / campaign_performance['EmailSent'] * 100).round(2)
campaign_performance['TotalCost'] = campaign_performance['EmailSent'] * campaign_performance['CostPerEmail']
campaign_performance['ROI%'] = ((campaign_performance['RevenueGenerated'] - campaign_performance['TotalCost']) / campaign_performance['TotalCost'] * 100).round(2)

print(campaign_performance[['EmailSent', 'OpenRate%', 'ClickRate%', 'ConversionRate%', 'RevenueGenerated', 'ROI%']])

# Overall Campaign KPIs
print(f"\n🎯 OVERALL CAMPAIGN KPIs:")
total_sent = campaign_df['EmailSent'].sum()
total_opened = campaign_df['EmailOpened'].sum()
total_clicked = campaign_df['EmailClicked'].sum()
total_converted = campaign_df['Converted'].sum()
total_revenue = campaign_df['RevenueGenerated'].sum()
total_cost = campaign_df['CostPerEmail'].sum()

print(f"   Emails Sent: {total_sent:,}")
print(f"   Open Rate: {(total_opened/total_sent*100):.2f}%")
print(f"   Click Rate: {(total_clicked/total_sent*100):.2f}%")
print(f"   Conversion Rate: {(total_converted/total_sent*100):.2f}%")
print(f"   Total Revenue: ${total_revenue:,.0f}")
print(f"   Total Cost: ${total_cost:,.0f}")
print(f"   Campaign ROI: {((total_revenue - total_cost) / total_cost * 100):.2f}%")

# ============================================================================
# SECTION 5: EXPORT DATA & INSIGHTS
# ============================================================================

print(f"\n📊 EXPORTING SEGMENT DATA...")

# Export Champion & Loyal customers for targeted campaigns
target_segments = rfm[rfm['Segment'].isin(['Champions', 'Loyal Customers'])][
    ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Segment']
].sort_values('Monetary', ascending=False)

target_segments.to_csv('high_value_customers.csv', index=False)
print(f"✓ Exported {len(target_segments)} high-value customers to 'high_value_customers.csv'")

# Export At-Risk customers
at_risk = rfm[rfm['Segment'] == 'At Risk'][
    ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Segment']
].sort_values('Recency')

at_risk.to_csv('at_risk_customers.csv', index=False)
print(f"✓ Exported {len(at_risk)} at-risk customers to 'at_risk_customers.csv'")

# ============================================================================
# SECTION 6: VISUALIZATIONS
# ============================================================================

print(f"\n📊 GENERATING VISUALIZATIONS...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

# Plot 1: Segment Distribution
ax1 = fig.add_subplot(gs[0, 0])
segment_counts = rfm['Segment'].value_counts().sort_values(ascending=True)
colors = plt.cm.Set3(np.linspace(0, 1, len(segment_counts)))
ax1.barh(segment_counts.index, segment_counts.values, color=colors, alpha=0.8)
ax1.set_xlabel('Number of Customers', fontsize=11, fontweight='bold')
ax1.set_title('Customer Segment Distribution', fontsize=12, fontweight='bold')
for i, v in enumerate(segment_counts.values):
    ax1.text(v + 50, i, str(int(v)), va='center', fontweight='bold')

# Plot 2: Average Monetary Value by Segment
ax2 = fig.add_subplot(gs[0, 1])
segment_monetary = rfm.groupby('Segment')['Monetary'].mean().sort_values(ascending=False)
colors_monetary = ['#27AE60' if x > 5000 else '#F39C12' if x > 2000 else '#E74C3C' for x in segment_monetary.values]
ax2.bar(range(len(segment_monetary)), segment_monetary.values, color=colors_monetary, alpha=0.7)
ax2.set_xticks(range(len(segment_monetary)))
ax2.set_xticklabels(segment_monetary.index, rotation=45, ha='right')
ax2.set_ylabel('Average Monetary Value ($)', fontsize=11, fontweight='bold')
ax2.set_title('Average Customer Value by Segment', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
for i, v in enumerate(segment_monetary.values):
    ax2.text(i, v + 100, f'${v:.0f}', ha='center', fontweight='bold', fontsize=9)

# Plot 3: Campaign Conversion Rate by Segment
ax3 = fig.add_subplot(gs[1, 0])
conversion_by_segment = campaign_df.groupby('Segment')['Converted'].apply(
    lambda x: (x.sum() / len(x) * 100)
).sort_values(ascending=False)
ax3.bar(range(len(conversion_by_segment)), conversion_by_segment.values, color='#3498DB', alpha=0.7)
ax3.set_xticks(range(len(conversion_by_segment)))
ax3.set_xticklabels(conversion_by_segment.index, rotation=45, ha='right')
ax3.set_ylabel('Conversion Rate (%)', fontsize=11, fontweight='bold')
ax3.set_title('Campaign Conversion Rate by Segment', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
for i, v in enumerate(conversion_by_segment.values):
    ax3.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=9)

# Plot 4: ROI by Segment
ax4 = fig.add_subplot(gs[1, 1])
roi_by_segment = campaign_performance['ROI%'].sort_values(ascending=False)
roi_colors = ['#27AE60' if x > 0 else '#E74C3C' for x in roi_by_segment.values]
ax4.barh(roi_by_segment.index, roi_by_segment.values, color=roi_colors, alpha=0.7)
ax4.set_xlabel('ROI (%)', fontsize=11, fontweight='bold')
ax4.set_title('Campaign ROI by Customer Segment', fontsize=12, fontweight='bold')
ax4.axvline(x=0, color='black', linestyle='--', linewidth=1)
for i, v in enumerate(roi_by_segment.values):
    ax4.text(v + 10 if v > 0 else v - 30, i, f'{v:.0f}%', va='center', fontweight='bold')

# Plot 5: RFM Score Distribution
ax5 = fig.add_subplot(gs[2, 0])
ax5.scatter(rfm['Frequency'], rfm['Monetary'], c=rfm['Recency'], 
           cmap='RdYlGn_r', alpha=0.6, s=50)
ax5.set_xlabel('Frequency', fontsize=11, fontweight='bold')
ax5.set_ylabel('Monetary Value ($)', fontsize=11, fontweight='bold')
ax5.set_title('Customer Distribution: Frequency vs Monetary (colored by Recency)', fontsize=12, fontweight='bold')
cbar = plt.colorbar(ax5.collections[0], ax=ax5)
cbar.set_label('Recency (days)', fontweight='bold')
ax5.grid(alpha=0.3)

# Plot 6: Revenue by Segment
ax6 = fig.add_subplot(gs[2, 1])
revenue_by_segment = campaign_df.groupby('Segment')['RevenueGenerated'].sum().sort_values(ascending=False)
colors_revenue = plt.cm.Greens(np.linspace(0.4, 0.8, len(revenue_by_segment)))
ax6.bar(range(len(revenue_by_segment)), revenue_by_segment.values, color=colors_revenue, alpha=0.8)
ax6.set_xticks(range(len(revenue_by_segment)))
ax6.set_xticklabels(revenue_by_segment.index, rotation=45, ha='right')
ax6.set_ylabel('Campaign Revenue Generated ($)', fontsize=11, fontweight='bold')
ax6.set_title('Campaign Revenue by Customer Segment', fontsize=12, fontweight='bold')
ax6.grid(axis='y', alpha=0.3)
for i, v in enumerate(revenue_by_segment.values):
    ax6.text(i, v + 1000, f'${v:.0f}', ha='center', fontweight='bold', fontsize=9)

plt.savefig('customer_segmentation_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Analysis dashboard saved as 'customer_segmentation_analysis.png'")

# ============================================================================
# SECTION 7: KEY RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 80)
print("KEY FINDINGS & MARKETING RECOMMENDATIONS")
print("=" * 80)

champion_count = len(rfm[rfm['Segment'] == 'Champions'])
at_risk_count = len(rfm[rfm['Segment'] == 'At Risk'])
lost_count = len(rfm[rfm['Segment'] == 'Lost'])

print(f"\n📊 KEY METRICS:")
print(f"   • Total Customers: {len(rfm):,}")
print(f"   • Champions: {champion_count:,}")
print(f"   • At Risk: {at_risk_count:,}")
print(f"   • Lost Customers: {lost_count:,}")
print(f"   • Campaign Average ROI: {((total_revenue - total_cost) / total_cost * 100):.2f}%")

print(f"\n💡 MARKETING RECOMMENDATIONS:")
print(f"   1. CHAMPIONS ({champion_count:,} customers)")
print(f"      → VIP treatment: exclusive offers, early access to new products")
print(f"      → Frequency: Monthly personalized communications")
print(f"      → Expected ROI Uplift: 20-30%")

print(f"\n   2. AT-RISK ({at_risk_count:,} customers)")
print(f"      → Win-back campaign: special discounts, appreciation offers")
print(f"      → Focus on: high-frequency touchpoints, service recovery")
print(f"      → Expected Retention Improvement: 15-25%")

print(f"\n   3. LOST ({lost_count:,} customers)")
print(f"      → Re-engagement: limited-time offers, product improvements")
print(f"      → Consider: lower-cost digital campaigns vs email")
print(f"      → Expected Re-activation Rate: 5-10%")

print(f"\n   4. OVERALL CAMPAIGN OPTIMIZATION:")
print(f"      → Focus budget on High-ROI segments (Champions, Loyal Customers)")
print(f"      → Reduce spend on Low-ROI segments (Lost, About to Sleep)")
print(f"      → Implement A/B testing for At-Risk segment messaging")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
