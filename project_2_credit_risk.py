"""
PROJECT 2: CREDIT RISK ASSESSMENT & LOAN PORTFOLIO ANALYSIS
===========================================================
Objective: Analyze loan portfolio performance, identify high-risk borrowers,
and provide insights for credit risk management.

Skills demonstrated: SQL, Python, Statistical Analysis, Risk Modeling, Visualization
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("=" * 80)
print("CREDIT RISK ASSESSMENT & LOAN PORTFOLIO ANALYSIS")
print("=" * 80)

# ============================================================================
# SECTION 1: CREATE LOAN DATABASE
# ============================================================================

print("\n📁 Setting up Loan Portfolio Database...")

# Create in-memory SQLite database
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

# Create Loans Table
cursor.execute('''
CREATE TABLE loans (
    LoanID TEXT PRIMARY KEY,
    BorrowerID TEXT,
    LoanAmount REAL,
    LoanTerm INT,
    InterestRate REAL,
    MonthlyPayment REAL,
    EmploymentLength INT,
    AnnualIncome REAL,
    DebtToIncomeRatio REAL,
    CreditScore INT,
    HomeOwnership TEXT,
    Purpose TEXT,
    State TEXT,
    IssueDate TEXT,
    Status TEXT
)
''')

# Create Transactions Table
cursor.execute('''
CREATE TABLE transactions (
    TransactionID TEXT PRIMARY KEY,
    LoanID TEXT,
    TransactionDate TEXT,
    AmountPaid REAL,
    Status TEXT,
    DaysLate INT,
    FOREIGN KEY (LoanID) REFERENCES loans(LoanID)
)
''')

# Generate synthetic loan data
np.random.seed(42)
n_loans = 5000

loans_data = []
for i in range(n_loans):
    loan_id = f"LN{i+1000:06d}"
    borrower_id = f"BW{i+100:06d}"
    loan_amount = np.random.uniform(5000, 500000)
    loan_term = np.random.choice([36, 60])
    credit_score = np.random.normal(700, 80)
    credit_score = np.clip(credit_score, 300, 850)
    
    # Interest rate based on credit score
    if credit_score >= 750:
        interest_rate = np.random.uniform(3.5, 6.0)
    elif credit_score >= 700:
        interest_rate = np.random.uniform(6.0, 10.0)
    elif credit_score >= 650:
        interest_rate = np.random.uniform(10.0, 14.0)
    else:
        interest_rate = np.random.uniform(14.0, 18.0)
    
    monthly_rate = interest_rate / 12 / 100
    monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**loan_term) / ((1 + monthly_rate)**loan_term - 1)
    
    employment_length = np.random.randint(0, 11)
    annual_income = np.random.uniform(25000, 250000)
    debt_to_income = (monthly_payment * 12) / annual_income
    
    # Determine loan status based on risk factors
    risk_score = 0
    if credit_score < 650:
        risk_score += 3
    elif credit_score < 700:
        risk_score += 2
    
    if debt_to_income > 0.4:
        risk_score += 3
    elif debt_to_income > 0.3:
        risk_score += 1
    
    if employment_length < 2:
        risk_score += 2
    
    # Status determination
    status_prob = risk_score / 10
    rand = np.random.random()
    if rand < status_prob * 0.3:
        status = 'Defaulted'
    elif rand < status_prob * 0.5:
        status = 'Late'
    elif rand < 0.9:
        status = 'Current'
    else:
        status = 'Paid Off'
    
    issue_date = (datetime.now() - timedelta(days=np.random.randint(30, 1095))).strftime('%Y-%m-%d')
    
    loans_data.append((
        loan_id, borrower_id, loan_amount, loan_term, interest_rate,
        monthly_payment, employment_length, annual_income, debt_to_income,
        int(credit_score), np.random.choice(['Rent', 'Own']), 
        np.random.choice(['Debt Consolidation', 'Auto', 'Personal', 'Home Improvement']),
        np.random.choice(['CA', 'NY', 'TX', 'FL', 'IL']), issue_date, status
    ))

cursor.executemany('''
INSERT INTO loans VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
''', loans_data)

# Generate transaction data
transaction_data = []
transaction_id_counter = 1000000
for loan in loans_data[:3000]:  # Only for active loans
    loan_id = loan[0]
    issue_date = datetime.strptime(loan[13], '%Y-%m-%d')
    
    # Generate multiple transactions
    current_date = issue_date
    for month in range(np.random.randint(1, 36)):
        current_date = current_date + timedelta(days=30)
        
        if current_date > datetime.now():
            break
        
        transaction_id = f"TX{transaction_id_counter}"
        transaction_id_counter += 1
        amount_paid = loan[5]  # monthly payment
        days_late = max(0, np.random.randint(-5, 60) if np.random.random() < 0.2 else 0)
        
        if days_late > 30:
            trans_status = 'Late'
        elif days_late > 0:
            trans_status = 'Late Payment'
        else:
            trans_status = 'On Time'
        
        transaction_data.append((
            transaction_id, loan_id, current_date.strftime('%Y-%m-%d'),
            amount_paid, trans_status, days_late
        ))

cursor.executemany('''
INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?)
''', transaction_data)

conn.commit()
print("✓ Database created with {} loans and {} transactions".format(n_loans, len(transaction_data)))

# ============================================================================
# SECTION 2: SQL ANALYSIS QUERIES
# ============================================================================

print("\n" + "=" * 80)
print("PORTFOLIO OVERVIEW & KPIs")
print("=" * 80)

# Query 1: Portfolio Summary
query1 = """
SELECT 
    COUNT(*) as TotalLoans,
    SUM(LoanAmount) as TotalLoanAmount,
    AVG(LoanAmount) as AvgLoanAmount,
    AVG(CreditScore) as AvgCreditScore,
    AVG(DebtToIncomeRatio) as AvgDebtToIncomeRatio
FROM loans
"""

summary = pd.read_sql_query(query1, conn)
print("\n📊 LOAN PORTFOLIO SUMMARY:")
print(f"   Total Loans: {int(summary['TotalLoans'].values[0]):,}")
print(f"   Total Amount Outstanding: ${summary['TotalLoanAmount'].values[0]:,.0f}")
print(f"   Average Loan Amount: ${summary['AvgLoanAmount'].values[0]:,.0f}")
print(f"   Average Credit Score: {summary['AvgCreditScore'].values[0]:.0f}")
print(f"   Average Debt-to-Income Ratio: {summary['AvgDebtToIncomeRatio'].values[0]:.2%}")

# Query 2: Loan Status Distribution
query2 = """
SELECT 
    Status,
    COUNT(*) as LoanCount,
    SUM(LoanAmount) as TotalAmount,
    ROUND(AVG(CreditScore), 0) as AvgCreditScore
FROM loans
GROUP BY Status
ORDER BY LoanCount DESC
"""

status_dist = pd.read_sql_query(query2, conn)
print("\n📈 LOAN STATUS DISTRIBUTION:")
print(status_dist.to_string(index=False))

# Query 3: Risk Assessment by Credit Score Bucket
query3 = """
SELECT 
    CASE 
        WHEN CreditScore >= 750 THEN 'Excellent (750+)'
        WHEN CreditScore >= 700 THEN 'Good (700-749)'
        WHEN CreditScore >= 650 THEN 'Fair (650-699)'
        ELSE 'Poor (<650)'
    END as CreditBucket,
    COUNT(*) as LoanCount,
    ROUND(SUM(LoanAmount), 0) as TotalAmount,
    ROUND(AVG(InterestRate), 2) as AvgInterestRate,
    ROUND(AVG(DebtToIncomeRatio), 3) as AvgDebtToIncomeRatio
FROM loans
GROUP BY CreditBucket
ORDER BY CreditScore DESC
"""

credit_analysis = pd.read_sql_query(query3, conn)
print("\n⭐ CREDIT SCORE ANALYSIS:")
print(credit_analysis.to_string(index=False))

# Query 4: Risk Metrics by Status
query4 = """
SELECT 
    Status,
    COUNT(*) as LoanCount,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM loans), 2) as PercentageOfPortfolio,
    ROUND(SUM(LoanAmount), 0) as TotalAmount,
    ROUND(AVG(CreditScore), 0) as AvgCreditScore,
    ROUND(AVG(DebtToIncomeRatio), 3) as AvgDTI
FROM loans
GROUP BY Status
"""

risk_metrics = pd.read_sql_query(query4, conn)
print("\n⚠️  RISK METRICS BY LOAN STATUS:")
print(risk_metrics.to_string(index=False))

# Query 5: Loan Purpose Analysis
query5 = """
SELECT 
    Purpose,
    COUNT(*) as LoanCount,
    ROUND(SUM(LoanAmount), 0) as TotalAmount,
    ROUND(AVG(CreditScore), 0) as AvgCreditScore,
    ROUND(100.0 * SUM(CASE WHEN Status IN ('Defaulted', 'Late') THEN 1 ELSE 0 END) / COUNT(*), 2) as DefaultRatePercent
FROM loans
GROUP BY Purpose
ORDER BY LoanCount DESC
"""

purpose_analysis = pd.read_sql_query(query5, conn)
print("\n📋 LOAN PURPOSE ANALYSIS:")
print(purpose_analysis.to_string(index=False))

# ============================================================================
# SECTION 3: ADVANCED RISK MODELING
# ============================================================================

print("\n" + "=" * 80)
print("ADVANCED RISK MODELING")
print("=" * 80)

# Load full dataset into pandas for advanced analysis
all_loans = pd.read_sql_query("SELECT * FROM loans", conn)

# Create Risk Score (0-100)
def calculate_risk_score(row):
    score = 0
    
    # Credit Score (0-40 points)
    if row['CreditScore'] < 600:
        score += 40
    elif row['CreditScore'] < 650:
        score += 30
    elif row['CreditScore'] < 700:
        score += 20
    elif row['CreditScore'] < 750:
        score += 10
    
    # Debt-to-Income Ratio (0-35 points)
    if row['DebtToIncomeRatio'] > 0.5:
        score += 35
    elif row['DebtToIncomeRatio'] > 0.4:
        score += 25
    elif row['DebtToIncomeRatio'] > 0.3:
        score += 15
    elif row['DebtToIncomeRatio'] > 0.2:
        score += 5
    
    # Employment Length (0-15 points)
    if row['EmploymentLength'] < 1:
        score += 15
    elif row['EmploymentLength'] < 2:
        score += 10
    elif row['EmploymentLength'] < 3:
        score += 5
    
    # Income Level (0-10 points)
    if row['AnnualIncome'] < 30000:
        score += 10
    elif row['AnnualIncome'] < 50000:
        score += 5
    
    return min(score, 100)

all_loans['RiskScore'] = all_loans.apply(calculate_risk_score, axis=1)

# Risk Categories
def categorize_risk(score):
    if score >= 70:
        return 'High Risk'
    elif score >= 40:
        return 'Medium Risk'
    else:
        return 'Low Risk'

all_loans['RiskCategory'] = all_loans['RiskScore'].apply(categorize_risk)

# Default Rate by Risk Category
print("\n🎯 DEFAULT RATE BY RISK CATEGORY:")
risk_default = all_loans.groupby('RiskCategory').apply(
    lambda x: pd.Series({
        'LoanCount': len(x),
        'DefaultCount': (x['Status'] == 'Defaulted').sum(),
        'DefaultRate%': (x['Status'] == 'Defaulted').sum() / len(x) * 100,
        'TotalAmount': x['LoanAmount'].sum(),
        'AvgRiskScore': x['RiskScore'].mean()
    })
).round(2)

print(risk_default.to_string())

# Expected Loss Calculation
print("\n💰 EXPECTED LOSS ANALYSIS:")
all_loans['ExpectedLoss'] = (
    (all_loans['Status'] == 'Defaulted').astype(int) * all_loans['LoanAmount'] * 0.5  # Assume 50% recovery
)

total_expected_loss = all_loans['ExpectedLoss'].sum()
portfolio_value = all_loans['LoanAmount'].sum()
loss_percentage = (total_expected_loss / portfolio_value) * 100

print(f"   Total Portfolio Value: ${portfolio_value:,.0f}")
print(f"   Expected Loss: ${total_expected_loss:,.0f}")
print(f"   Loss as % of Portfolio: {loss_percentage:.2f}%")

# ============================================================================
# SECTION 4: EXPORT HIGH-RISK LOANS
# ============================================================================

print("\n📋 EXPORTING HIGH-RISK LOANS...")

high_risk_loans = all_loans[all_loans['RiskCategory'] == 'High Risk'][
    ['LoanID', 'BorrowerID', 'LoanAmount', 'CreditScore', 'DebtToIncomeRatio', 
     'RiskScore', 'Status', 'InterestRate']
].sort_values('RiskScore', ascending=False)

high_risk_loans.to_csv('high_risk_loans.csv', index=False)
print(f"✓ Exported {len(high_risk_loans)} high-risk loans to 'high_risk_loans.csv'")

# ============================================================================
# SECTION 5: VISUALIZATIONS
# ============================================================================

print("\n📊 GENERATING VISUALIZATIONS...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Plot 1: Loan Status Distribution
ax1 = fig.add_subplot(gs[0, 0])
status_counts = all_loans['Status'].value_counts()
colors = ['#27AE60', '#F39C12', '#E74C3C', '#95A5A6']
ax1.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', 
        colors=colors, startangle=90)
ax1.set_title('Loan Status Distribution', fontsize=12, fontweight='bold')

# Plot 2: Risk Category Distribution
ax2 = fig.add_subplot(gs[0, 1])
risk_counts = all_loans['RiskCategory'].value_counts()
risk_order = ['Low Risk', 'Medium Risk', 'High Risk']
risk_counts = risk_counts.reindex(risk_order)
risk_colors = ['#27AE60', '#F39C12', '#E74C3C']
ax2.bar(risk_counts.index, risk_counts.values, color=risk_colors, alpha=0.7)
ax2.set_ylabel('Number of Loans', fontsize=11, fontweight='bold')
ax2.set_title('Risk Category Distribution', fontsize=12, fontweight='bold')
for i, v in enumerate(risk_counts.values):
    ax2.text(i, v + 50, str(int(v)), ha='center', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Default Rate by Risk Category
ax3 = fig.add_subplot(gs[1, 0])
risk_default_rate = all_loans.groupby('RiskCategory')['Status'].apply(
    lambda x: (x == 'Defaulted').sum() / len(x) * 100
).reindex(risk_order)
ax3.bar(risk_default_rate.index, risk_default_rate.values, color=risk_colors, alpha=0.7)
ax3.set_ylabel('Default Rate (%)', fontsize=11, fontweight='bold')
ax3.set_title('Default Rate by Risk Category', fontsize=12, fontweight='bold')
for i, v in enumerate(risk_default_rate.values):
    ax3.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Credit Score vs Loan Amount
ax4 = fig.add_subplot(gs[1, 1])
scatter = ax4.scatter(all_loans['CreditScore'], all_loans['LoanAmount'], 
                      c=all_loans['RiskScore'], cmap='RdYlGn_r', alpha=0.6, s=30)
ax4.set_xlabel('Credit Score', fontsize=11, fontweight='bold')
ax4.set_ylabel('Loan Amount ($)', fontsize=11, fontweight='bold')
ax4.set_title('Credit Score vs Loan Amount (colored by Risk Score)', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax4, label='Risk Score')
ax4.grid(alpha=0.3)

# Plot 5: Interest Rate Distribution by Status
ax5 = fig.add_subplot(gs[2, 0])
status_labels = ['Current', 'Paid Off', 'Late', 'Defaulted']
for status in status_labels:
    data = all_loans[all_loans['Status'] == status]['InterestRate']
    if len(data) > 0:
        ax5.hist(data, bins=20, alpha=0.5, label=status)
ax5.set_xlabel('Interest Rate (%)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax5.set_title('Interest Rate Distribution by Loan Status', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

# Plot 6: Loan Purpose Default Rates
ax6 = fig.add_subplot(gs[2, 1])
purpose_default = all_loans.groupby('Purpose')['Status'].apply(
    lambda x: (x == 'Defaulted').sum() / len(x) * 100
).sort_values(ascending=False)
ax6.barh(purpose_default.index, purpose_default.values, color='#E74C3C', alpha=0.7)
ax6.set_xlabel('Default Rate (%)', fontsize=11, fontweight='bold')
ax6.set_title('Default Rate by Loan Purpose', fontsize=12, fontweight='bold')
for i, v in enumerate(purpose_default.values):
    ax6.text(v + 0.1, i, f'{v:.1f}%', va='center', fontweight='bold')
ax6.grid(axis='x', alpha=0.3)

plt.savefig('loan_portfolio_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Analysis dashboard saved as 'loan_portfolio_analysis.png'")

# ============================================================================
# SECTION 6: KEY RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 80)
print("KEY FINDINGS & RISK MANAGEMENT RECOMMENDATIONS")
print("=" * 80)

total_defaults = (all_loans['Status'] == 'Defaulted').sum()
default_rate = total_defaults / len(all_loans) * 100

print(f"\n📊 KEY METRICS:")
print(f"   • Total Loans Analyzed: {len(all_loans):,}")
print(f"   • Total Defaulted: {total_defaults:,} ({default_rate:.2f}%)")
print(f"   • High-Risk Loans: {len(all_loans[all_loans['RiskCategory'] == 'High Risk']):,}")
print(f"   • Expected Loss: ${total_expected_loss:,.0f}")

print(f"\n💡 RECOMMENDATIONS:")
print(f"   1. Implement enhanced monitoring for {len(high_risk_loans)} high-risk loans")
print(f"   2. Increase collateral requirements for scores < 650")
print(f"   3. Reduce loan amounts for DTI > 0.4")
print(f"   4. Add provisions for loans in default status")
print(f"   5. Tighten underwriting criteria for {purpose_analysis.iloc[purpose_analysis['DefaultRatePercent'].idxmax()]['Purpose']} loans")

conn.close()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
