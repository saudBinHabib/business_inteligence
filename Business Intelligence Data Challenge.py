#!/usr/bin/env python
# coding: utf-8

# # Business Intelligence Data Challenge

# In[ ]:





# ## Data Set Insights

# In[ ]:





# #### table_A_conversions.csv:
# 
#  - example list of conversions/ sales/ transaction
#  - Conv_ID - transaction ID
#  - Conv_Date - transaction date
#  - Revenue - value of transaction
#  - User_ID - an ID of a customer
# 
# #### table_B_attribution.csv:
# 
#  - list of attribution results for conversions
#  - Conv_ID - transaction ID (link to table A)
#  - Channel - marketing channel
#  - IHC_Conv - attributed conversion fraction by the IHC model
# 
# #### Note, that the attributed conversion fraction (IHC_Conv), i.e. it is the fraction of the given conversion which is attributed to a certain channel, sums up to 1.0 for every conversion.

# In[ ]:





# ## Notebook Structure.

# In[ ]:





# #### 1. Importing Dataset
# #### 2. Insights about the Datasets.
# 	- Insights
# 	- let's look for the completeness of the both datasets.
# #### 3. Data Wrangling.
# 	- Lets's Combine the both dataset for getting the information of the Channels and IHC_Conv, and insights about the dataset, after combining.
# 	- Changing the Columns Data Types.
# 	- Facts from the Conversions Dataset.
# 	- Facts from the the Dataset after mergin both.
# 	- Comparison.
# #### 4. RFM Segmentation
# 	- RfmSegmentation according to the Conversions.
# 	- Interpretation.
# 	- Quartile Method.
# 	- Evaluation of the Quartile Method.
# 	- checking the conversion of the top User, according to the RFM and Quartile Method.
# #### 5. Cohort Analysis
# 	- Calculating Cohort.
# 	- Interpretation.
# 	- Results.
# 	- Plotting the Cohort Analysis Output.
# 

# In[ ]:





# In[1]:


# Importing all important packages.


# In[1]:


import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

import datetime as dt                            # for creating datetime variable
import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
import seaborn as sns                            # more plots

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# # 1) Importing the dataset.

# In[ ]:





# In[2]:


# importing teh Conversions dataset.
df_conversions = pd.read_csv('dataset/table_A_conversions.csv')

# importing the attributions dataset.
df_attributions = pd.read_csv('dataset/table_B_attribution.csv')


# In[ ]:





# # 2) Insights About Datasets.

# In[ ]:





# ### i) Insights

# In[3]:


# checking the shape of conversion dataset. 
print('Shape of First: ', df_conversions.shape)

# checking the shape of attribution dataset.
print('Shape of Second: ', df_attributions.shape)


# In[4]:


# checking the top 5 rows of converstion dataframe .

df_conversions.head(5)


# In[5]:


# checking the top 5 rows of attribution dataframe.

df_attributions.head(5)


# In[ ]:




From the above, we can see that we have CONV_ID, which is same in the both dataframes and we can merge the both datasets using the CONV_ID, to get the Channels and IHC_Conv
# In[ ]:





# ### ii) let's look for the completeness of the both datasets.

# In[6]:


# lets check the info for the converstions dataframe.

df_conversions.info()


# In[7]:


# check the columns of the conversions dataset, if they have any null values.

print(df_conversions.isnull().sum().sort_values(ascending=False))


# In[ ]:





# In[8]:


# lets check the info for the second daaframe.

df_attributions.info()


# In[9]:


# check the columns of the attributions dataset, if they have any null values.

print(df_attributions.isnull().sum().sort_values(ascending=False))


# In[ ]:





# In[ ]:





# # 3) Data Wrangling

# ### i) Lets's Combine the both dataset for getting the information of the Channels and IHC_Conv, and insights about the dataset, after combining.

# In[47]:


# let's merge the dataset using Conv_ID

df_merged = pd.merge(df_conversions, df_attributions, how='inner', on='Conv_ID',left_index=False, right_index=False, sort=True)


# In[48]:


# checking the shape of the dataset, after merging.

df_merged.shape


# In[49]:


# checking the top 5 rows of the dataset.

df_merged.head()


# In[50]:


# checking the completeness of the dataset.

print(df_merged.isnull().sum().sort_values(ascending=False))


# In[ ]:





# ### ii) Changing the Columns Data Types.

# In[ ]:





# In[51]:


# Changing the data type of the Conv_Date Attribute

df_merged['Conv_Date'] = pd.to_datetime(df_merged['Conv_Date'])


# In[52]:


# changing the data type of Conv_Date attribute in actual conversion dataset.

df_conversions['Conv_Date'] = pd.to_datetime(df_conversions['Conv_Date'])


# In[ ]:





# ### iii) Facts from the Conversions Dataset. 

# In[ ]:





# In[16]:


# Let's check some stats about our conversions dataframes.

print("Inforation about our Conversions dataset \n\n")
print("Number of Rows   \t\t:", df_conversions.shape[0]) #check the total rows in our data.
print("Number of Columns \t\t:", df_conversions.shape[1]) #check the total coloumns in our data.
print("Date ranges from  \t\t:", df_conversions.Conv_Date.min(), " to ", df_conversions.Conv_Date.max()) #check range of the dates in the pandas.
print("#Number of Unique Conv_ID \t:", df_conversions.Conv_ID.nunique()) #check the Number of Unique transactions
print("#Unique Customers \t\t:", df_conversions.User_ID.nunique()) #check the unique customers
print("Range of Revenue  \t\t:", df_conversions.Revenue.min(), " to ", df_conversions.Revenue.max()) #check range Quantity pada data # check the unique channels in our dataste.


# In[ ]:





# ### iv) Facts from the the Dataset after mergin both.

# In[ ]:





# In[53]:


# Let's check some stats about merged dataset.

print("Inforation about our Complete dataset \n\n")
print("Number of Rows   \t\t:", df_merged.shape[0]) #check the total rows in our data.
print("Number of Columns \t\t:", df_merged.shape[1]) #check the total coloumns in our data.
print("Date ranges from  \t\t:", df_merged.Conv_Date.min(), " to ", df_merged.Conv_Date.max()) #check range of the dates in the pandas.
print("#Number of Unique Conv_ID \t:", df_merged.Conv_ID.nunique()) #check the Number of Unique transactions
print("#Unique Customers \t\t:", df_merged.User_ID.nunique()) #check the unique customers
print("Range of Revenue  \t\t:", df_merged.Revenue.min(), " to ", df_merged.Revenue.max()) #check range Quantity pada data
print("Range of IHC Value \t\t:", df_merged.IHC_Conv.min(), " to ", df_merged.IHC_Conv.max()) #check range of IHC values.
print("Unique Channels in Our Dataset  :", df_merged.Channel.unique().tolist()) # check the unique channels in our dataste.


# In[ ]:





# ### v) Comparison

# In[ ]:





# In[54]:


# Checking the stats of the Numeric columns of our dataset. 

df_merged.describe()


# In[19]:


# checking the stats of Numeric Values of Convesions Dataset

df_conversions.describe()


# ## Note:
# 
# #### Frome Above we can see that, after joining the dataset, we have too many repeating conversion_ids, which cause the change in Reveune and the frequency of a customer shopping at our store. So for the sake of further analysis, we will gonna use "df_conversions" because it have the actual revenue, total number of conversions and Actual Dates on which conversion happened.

# In[ ]:





# #  Data Analysis. 

# In[ ]:





# ## 4) RFM Segmentation

# RFM Segmentation is customer segmentation based on scoring R, F, and M (Recency: Length of day since the last transaction, Frequency: Number of transactions, Monetary: Total Revenue).
# 
# Because the last transaction on the data was March 26, 2018, we will use March 27, 2018 to calculate the recency

# In[ ]:





# ### i) RfmSegmentation according to the Conversions.

# In[ ]:





# In[20]:


# creating a date time value for March 27, 2018

NOW = dt.datetime(2018, 3, 27)

# creating a rfmTransTable by aggregating the date, conv_id, and revenue.
rfmTransTable = df_conversions.groupby('User_ID').agg({'Conv_Date': lambda x: (NOW - x.max()).days, 'Conv_ID': lambda x: len(x), 'Revenue': lambda x: x.sum()})

# changing the data type of the date column
rfmTransTable['Conv_Date'] = rfmTransTable['Conv_Date'].astype(int)

# renaming the column names of rfmTransTable
rfmTransTable.rename(columns = {'Conv_Date': 'recency',
                         'Conv_ID': 'frequency',
                         'Revenue': 'monetary'}, inplace=True)


# In[21]:


# sorting the rfmTransTable in descending order

rfmTransTable = rfmTransTable.sort_values(by=['frequency'], ascending=False)


# In[22]:


# Checking the Top 10 Rows

rfmTransTable.head(10)


# In[ ]:





# ### ii) Interpretation

# In[ ]:




User with ID 2c75940486d75040f269c9671ab746dffefe9692 have frequency : 111 (111 time transaction), recency : 4 (4 days of the last transaction), and monetary 29117.30312 (Grand total transactions)  

user with ID 8a1846b853d9522214daba775a46789ada386c3a has frequency : 41 (41 complete transaction times), recency : 1 (1 days from the last transaction), and monetary 4863.15544 (Grand total transactions)

The easiest way to create a segment is the quartile method. With this method there will be 4 segments that are easy to understand
# In[ ]:





# ### iii) Quartile Method

# In[ ]:





# In[23]:


# Making quantile from the rfmTransTable

quantiles = rfmTransTable.quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()
segmented_rfm = rfmTransTable


# In[24]:


# checking the top rows of segmented_rfm

segmented_rfm.head()


# In[25]:


# method for calculating the R Score.
def RScore(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1

# method for calculating the FM score.
def FMScore(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4


# In[26]:


# calculating the r_quartile using Rscore method.
segmented_rfm['r_quartile'] = segmented_rfm['recency'].apply(RScore, args=('recency',quantiles,))

# calculating the f_quartile using FMScore method.
segmented_rfm['f_quartile'] = segmented_rfm['frequency'].apply(FMScore, args=('frequency',quantiles,))

# calculating the m_quartile using FMScore method.
segmented_rfm['m_quartile'] = segmented_rfm['monetary'].apply(FMScore, args=('monetary',quantiles,))

# checking the top 5 rows of segmented_rfm
segmented_rfm.head()


# In[29]:


# calculating the RFM Segment by combining all the quartiles as a string.
segmented_rfm['RFMSegment'] = segmented_rfm.r_quartile.map(str) + segmented_rfm.f_quartile.map(str) + segmented_rfm.m_quartile.map(str)

# calculating the RFM Segment by combining all the quartiles as a string.
segmented_rfm['RFMScore'] = segmented_rfm.r_quartile + segmented_rfm.f_quartile + segmented_rfm.m_quartile

# checking the top 10 rows of segmented_rfm
segmented_rfm.head(10)


# In[28]:


# checking the last 5 records of the segmented_rfm

segmented_rfm.tail()


# In[ ]:





# ### iv) Evaluation of the Quartile Method.

# In[ ]:




RFM Score: 12 is the best score because it has a low recency (still active), frequency high (often making transactions) and monetary high.
# In[ ]:





# In[30]:


# checking the top RFMScore, by sorting the dataframe according to monetary in descending order.

segmented_rfm[segmented_rfm['RFMScore'] == 12].sort_values('monetary', ascending=False).head(10)


# In[45]:


# ploting Histogram of the RFM Scores.

segmented_rfm.RFMScore.hist()


# ### v)  checking the conversion of the top User, according to the RFM and Quartile Method. 

# In[ ]:





# In[31]:


# checking the conversion of the top User.

# filtering to get the conversion of the top user with USER_ID
top_customer = df_conversions[df_conversions['User_ID'] == '2c75940486d75040f269c9671ab746dffefe9692']

# checking first 10 conversions.
top_customer.head(10)


# In[ ]:





# # 5) Cohort Analysis

# In[ ]:




Cohort Analysis are used for calculating the Retention rates. Retention rates are very important for the success of any business, but somehow Retention rates are often ignored.

The Reason why Retention Rates are very important. Because the cost of customer acquisition is very expensive we need to do everything to convince the client to return after their first purchase.

If your retention rate is low you will spend a budget for the acquisition channel so that more customers will arrive.

From Cohort Analysis we can see the retention rate or what percentage of customers return in the following months after the first purchase


# In[ ]:





# In[32]:


# checking the head of conversions dataset.

df_conversions.head()


# In[ ]:





# ### i) Calculating Cohort

# In[ ]:





# In[33]:


# declaring a method to get us the first date of each month from the dataset.
def get_month(x): return dt.datetime(x.year, x.month, 1)

# creating a column of Conv_Month which can hold the first date of each month
df_conversions['Conv_Month'] = df_conversions['Conv_Date'].apply(get_month)

# create a grouping dataframe by grouping the user_id according to the months.
grouping = df_conversions.groupby('User_ID')['Conv_Month']

# create a CohortMonth column in the conversions database with the minimum value.
df_conversions['CohortMonth'] = grouping.transform('min')


# In[34]:


# checking the first 5 rows of the conversions dataset to have a look.

df_conversions.head()


# In[35]:


## function for extracting the year, month and day from the date value. 

def get_date_int(df, column):
    year = df[column].dt.year
    month = df[column].dt.month
    day = df[column].dt.day
    return year, month, day


# In[36]:


# getting the year and month of Conv_Month Column from the get_date_int method.
invoice_year, invoice_month, _ = get_date_int(df_conversions, 'Conv_Month')

# getting the year and month of CohortMonth Column from the get_date_int method.
cohort_year, cohort_month, _ = get_date_int(df_conversions, 'CohortMonth')


# In[37]:


# calculating the years difference among the invoice and cohort year.
years_diff = invoice_year - cohort_year

# calculating the month difference among the invoice and cohort month.
months_diff = invoice_month - cohort_month


# In[38]:


# creating the cohortIndex.

df_conversions['CohortIndex'] = years_diff * 12 + months_diff + 1


# In[39]:


# checking the top 5 rows of conversion dataframe.

df_conversions.head()


# In[40]:


# checking the null values in our conversions dataset.

df_conversions.isna().sum()


# In[ ]:





# In[41]:


## grouping the dataset according to the CohortMonth and CohortIndex
grouping = df_conversions.groupby(['CohortMonth', 'CohortIndex'])

# creating a cohort data by getting unique User_ID
cohort_data = grouping['User_ID'].apply(pd.Series.nunique)

# reseting the index of the data.
cohort_data = cohort_data.reset_index()

# creating the cohort counts by making cohortmonth index and cohortIndex as column and User_ID as value of those columns
cohort_counts = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='User_ID')


# In[42]:


# checking the cohort_counts dataset.

cohort_counts


# In[ ]:





# ### ii) Interpretation 

# In[ ]:




CohortMonth 2017-03 (Cohort March 2017) has 4448 Unique Users who made conversions that month (CohortIndex 1),
1047 customers returned to the transaction the following month (CohortIndex 2),
637 customers returned again the following month (CohortIndex 3), and so on....
# In[ ]:





# ### iii) Results

# In[ ]:





# In[43]:


# Assigning first column as cohort size.
cohort_sizes = cohort_counts.iloc[:,0]

# calulcating the retention by dividing with the cohort size.
retention = cohort_counts.divide(cohort_sizes, axis=0) 
retention.round(2) * 100


# In[ ]:





# ### iv) Plotting the Cohort Analysis Output.

# In[ ]:





# In[44]:


# Ploting the Cohort Analysis

# defining the fig size.
plt.figure(figsize=(15, 8))

# assigning the plot title.
plt.title('Retention rates')

# ploting the chart usign Seaborn.
sns.heatmap(
    data = retention,
    annot = True,
    fmt = '.0%',
    vmin = 0.0,
    vmax = 0.5,
    cmap = 'GnBu_r'
)

# Showing the plot
plt.show()


# In[ ]:





# In[ ]:




