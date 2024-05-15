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

male_count = df_encouded['gender_Male'].sum()
female_count = df_encouded['gender_Female'].sum()
labels = ['male', 'female']
sizes = [male_count,female_count]
colors = ['blue', 'pink']
plt.figure(figsize = (6,6))
plt.pie(sizes, colors=colors, labels = labels, startangle=50, autopct='%1.1f%%') #labels,colors, autopct='%1.1f%')
plt.title('gender dist')
#Pie chart clearly indicates that the majority of respondents in the dataset are males,
#significantly outnumbering females

table = pd.crosstab(index = df['gender'], columns = df['Occupation'])
table.plot(kind = 'bar', stacked = True)
plt.xticks(rotation = 20)
#Stacked bar chart clearly illustrates that among investors, the salaried occupation is the
#most common for both males and females

table = pd.crosstab(index = df['Purpose_investment'], columns = df['Factors_investment'])
table.plot(kind = 'bar', stacked = True)
plt.xticks(rotation = 20)
#In the stacked bar chart that among investors, the most important purpose for
#investment is wealth creation and the important factor influencing this decision is the
#expected return on investment

numeric_columns=[]
for column in df.columns:
    if df[column].dtype=='int64':
        numeric_columns.append(column)
        
corr_matrix= df[numeric_columns].corr()
# Using for loop here eliminates the necessity of hardcoding column names and rather it
# easier to handle dataset modifications

sns.heatmap(corr_matrix, cmap = 'coolwarm', annot=True)
plt.title('corr Heatmap')
# G-secs and Corporate bonds show a noticeable positive relationship with a correlation
# coefficient of 0.61 on the correlation matrix

plt.figure(figsize=(4,4))
colormap = plt.cm.coolwarm
plt.scatter(df['Age'], df['Mutual_funds'], cmap = colormap)
plt.xlabel('Age')
plt.ylabel('Mutual funds')
plt.title('Scatter plot: age vs. MF')
plt.axhline(y = 7, color = 'green', linestyle = '--', label = 'Rank 7')
plt.axvline(x = 20, color = 'red', linestyle = '--',label = 'age 20')
plt.axvline(x = 39, color = 'red', linestyle = '--',label = 'age 39')
plt.colorbar(label = 'Rank')
plt.legend()
# It appears that investors in the age group of 20 to 39 consistently assigned the highest
# ranking 7 to mutual funds, indicating strong preference for this investment avenue

plt.figure(figsize=(4,4))
colormap = plt.cm.coolwarm
plt.scatter(df['Age'], df['Equity_market'], cmap = colormap)
plt.xlabel('age')
plt.ylabel('Equity')
plt.title('scatter plot: age vs. equity')
plt.axhline(y = 5,color = 'green', linestyle = '--', label = 'Rank5')
plt.axhline(y = 7,color = 'green', linestyle = '--', label = 'Rank7')
plt.axvline(x = 27 ,color = 'red', linestyle = '--', label = 'Age27')
plt.axvline(x = 38,color = 'red', linestyle = '--', label = 'Age38')
plt.colorbar(label = 'Rank')
plt.legend()
# It appears that investors in the age group of 27 to 38 consistently assigned the highest
# ranking 7 to equity, indicating strong preference for this investment avenue

plt.figure(figsize=(4,4))
colormap = plt.cm.coolwarm
plt.scatter(df['Age'], df['Corporate_bonds'], cmap = colormap)
plt.xlabel('age')
plt.ylabel('Corprate Bonds')
plt.title('scatter plot: age vs. corporare bonds')
plt.axhline(y = 4,color = 'green', linestyle = '--', label = 'Rank4')
plt.axvline(x = 21 ,color = 'red', linestyle = '--', label = 'Age21')
plt.axvline(x = 37,color = 'red', linestyle = '--', label = 'Age37')
plt.colorbar(label = 'Rank')
plt.legend()
# Investors falling within the age range of 21 to 37 predominantly assign a ranking of 4 to
# corporate bonds, suggesting that this group considers corporate bonds as moderately
# important in their investment choices

plt.figure(figsize=(4,4))
colormap = plt.cm.coolwarm
plt.scatter(df['Age'], df['G_secs'], cmap = colormap)
plt.xlabel('age')
plt.ylabel('G secs')
plt.title('scatter plot: age vs. G_secs')
plt.axhline(y = 3 ,color = 'green', linestyle = '--', label = 'Rank3')
plt.axvline(x = 23 ,color = 'red', linestyle = '--', label = 'Age20')
plt.axvline(x = 35,color = 'red', linestyle = '--', label = 'Age40')
plt.colorbar(label = 'Rank')
plt.legend()
# Investors aged 23 to 35 generally consider government securities with a ranking of 3,
# signifying moderate importance

plt.figure(figsize=(4,4))
colormap = plt.cm.coolwarm
plt.scatter(df['Age'], df['FD'],c= df['FD'], cmap = colormap, marker = 'o')
plt.xlabel('age')
plt.ylabel('fixed deposite')
plt.title('scatter plot: age vs. FD')
plt.axhline(y = 5 ,color = 'green', linestyle = '--', label = 'Rank5')
plt.axvline(x = 23 ,color = 'red', linestyle = '--', label = 'Age20')
plt.axvline(x = 39,color = 'red', linestyle = '--', label = 'Age40')
plt.colorbar(label = 'Rank')
plt.legend()

# Investors aged 23 to 39 commonly prioritize fixed deposits, assigning them a ranking of
# 5, indicating a moderate level of importance

plt.figure(figsize=(4,4))
colormap = plt.cm.coolwarm
plt.scatter(df['Age'], df['PPF'],c= df['PPF'], cmap = colormap, marker = 'o')
plt.xlabel('age')
plt.ylabel('Public Provident Fund')
plt.title('scatter plot: age vs.ppf')
plt.axhline(y =7 ,color = 'green', linestyle = '--', label = 'Rank7')
plt.axvline(x = 22 ,color = 'red', linestyle = '--', label = 'Age20')
plt.axvline(x = 40,color = 'red', linestyle = '--', label = 'Age40')
plt.colorbar(label = 'Rank')
plt.legend()

#Investors between the ages of 22 to 40, the majority of investors ranked PPF as highly
#important 7 in their investment decisions

plt.figure(figsize=(4,4))
colormap = plt.cm.coolwarm
plt.scatter(df['Age'], df['Gold/SGB'],c= df['Gold/SGB'], cmap = colormap, marker = 'o')
plt.xlabel('age')
plt.ylabel('Gold/SGB')
plt.title('scatter plot: age vs.ppf')
plt.axhline(y =7 ,color = 'green', linestyle = '--', label = 'Rank7')
plt.axhline(y =6 ,color = 'lime', linestyle = '--', label = 'Rank6')
plt.axhline(y =3 ,color = 'pink', linestyle = '--', label = 'Rank3')
plt.axvline(x = 23 ,color = 'red', linestyle = '--', label = 'Age20')
plt.axvline(x = 35,color = 'red', linestyle = '--', label = 'Age40')
plt.colorbar(label = 'Rank')
plt.legend()

#Gold/SGB received the ranking 7 from investors in the age group of 23 to 35
