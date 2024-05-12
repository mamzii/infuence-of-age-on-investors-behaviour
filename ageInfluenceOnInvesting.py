## Import the required libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
# In[1]
df = pd.read_csv('M:/tutorials/DATA SCIENCE/influence of age on investors behavior/investors.csv')
print(df.shape)
# In[2]
# Column headers require data preprocessing due to their length.
# Need to drop the unnecessary columns to focus on the relevant data.
df.drop(columns = ['Timestamp','Username'], axis = 1, inplace = True)
# drop timestamp(Useless) and Username(all is null)

# rename column(make the data organized)
columns_rename_mapping = {
    'Which best describes your gender?':'gender',
    'What is your occupation?':'Occupation',
    'What is your highest education level?':'Education_level',
    'Do you invest in Investment Avenues?':'Investment_avenues',
    'What proportion of money you invest?':'Proportion_invest',
    "What do you think are the best options for investing your money? (Rank in order of preference) [Mutual Funds]":'Mutual_funds',
    "What do you think are the best options for investing your money? (Rank in order of preference) [Equity Market]":'Equity_market',
    "What do you think are the best options for investing your money? (Rank in order of preference) [Corporate Bonds]":'Corporate_bonds',
    "What do you think are the best options for investing your money? (Rank in order of preference) [Government Bonds]":"G_secs",
    "What do you think are the best options for investing your money? (Rank in order of preference) [Fixed Deposits]":'FD',
    "What do you think are the best options for investing your money? (Rank in order of preference) [PPF - Public Provident Fund]":'PPF',
    "What do you think are the best options for investing your money? (Rank in order of preference) [Gold / Sovereign Gold Bonds - SGB]":'Gold/SGB',
    'Do you invest in Stock market?':'Invest_stocks',
    'What are the factors considered by you while investing in any instrument?':'Factors_investment',
    'What is your investment objective?':'Investment_objective',
    'How long do you prefer to keep your money in any investment instrument?':'Duration',
    'How often do you monitor your investments?':'Investment_monitor',
    'How much return do you expect from any investment instrument?':'Expected_return',
    'Which investment avenue do you mostly invest in?':'Preferred_avenue',
    'What are your savings objectives?':'Savings_objective',
    'Reasons for investing in Equity Market':'Reason_equity',
    'Reasons for investing in Mutual Funds':'Reason_MF',
    'What is your purpose behind investment?':'Purpose_investment',
    'Reasons for investing in Government Bonds':'Reason_Gsec',
    'Reasons for investing in Fixed Deposits':'Reason_FD',
    'Your major source of information for investment is':'Source'
    }
df.rename(columns = columns_rename_mapping, inplace=True)

# get the head of data frame
df.head()

# get the tail fo data frame
df.tail()

## Concise summary of Data Frame
df.info()

# check Na values
df.isna().sum()

# Checking duplicate values
df.duplicated().sum()

# The dataset is ready
# In[]
df.describe()
# it works on numeric columns so we consider age column
# Count: 132 responses
# Mean: 32.99 or 33 years old
# Standard Deviation: 10.92 years old, suggests ages of the people in the dataset are not very close to average age i.e. 32.99
# Minimum: 18 years old
# 25th Percentile: 25.75 years old (25% of peaple is below 25.75)
# Median (50th Percentile): 30 years old
# 75th Percentile: 37.25 years old
# Maximum: 70 years old

