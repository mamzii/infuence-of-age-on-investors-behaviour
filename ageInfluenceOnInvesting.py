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

df.hist(figsize = (10,10), rwidth = 0.95, color = 'skyblue', grid = False)
plt.title('Distribbution')
# Age histogram representation appears to be right-skewed(fall positive side of the peak).
# Other histogram have values ranging from 1 to 7 which exhibit no skewness.

plt.figure(figsize = (6,6))
sns.histplot(df['Age'], kde = True)
plt.title('Age Distribution')

# kde plot
#sns.kdeplot(data=df, x='Age')
# Age distribution is right-skewed. Let's understand it this way, it means that most people in the group are younger and there are very few older people in the data set. Imagine seesaw where younger > older people.

plt.figure(figsize=(6,6))
sns.boxplot(data = df)                              # in ref code it was like this(sns.boxplot(df)) that the python raise an error
plt.xticks(rotation=40)
plt.title('Box-plot of numeric columns')

# Box plot shows that only the Age column has outliers, while the other numeric columns do not

plt.figure(figsize = (4,4))
sns.boxplot(data = df, y = 'Age')
plt.title('boxplot of Age')
# Due to the small size, I can easily identify 7 outliers, primarily in the Age column

## Creating an outliers function for calculating outliers in dataset
def outliers():
    Q1=df['Age'].quantile(0.25) ## 1st quartile is 25.75
    Q2=df['Age'].quantile(0.5) ## 2nd quartile is 30 i.e. meadian
    Q3=df['Age'].quantile(0.75) ## 3rd quartile is 37.25
    IQR=Q3-Q1 ## Inter quartile range is 11.5
    lower_bound=Q1-1.5*IQR ## Lower whisker is 8.5
    upper_bound=Q3+1.5*IQR ## Upper whisker is 54.5
    ## Anything above the upper_bound and below the lower_bound becomes the outliers
    return df[(df['Age']<lower_bound) | (df['Age']>upper_bound)]

## Calling outlier function
outliers()

plt.figure(figsize = (6,6))
sns.boxplot(data = df, x = 'gender', y = 'Age')
plt.title('Age dist by gender')
plt.xlabel('gender')
plt.ylabel('age')
plt.xticks(rotation = 30)
# After conducting categorical and numerical analysis comparing Gender and Age using box plots, it is discovered that 
#there are 6 outliers in the male and 1 outlier in the female category. But in initial analysis, only male outliers were observed

sns.countplot(data = df, x = 'Occupation', palette = 'Set1',edgecolor = 'black')
plt.title('occupation dist')
plt.xlabel('occupation')
# Salaried investors dominate the dataset in terms of occupation

df_encouded = pd.get_dummies(df, columns=['gender'])

#The Gender column has been one-hot encoded, this will result two new columns i.e.
# Gender_Male and Gender_Female o
























