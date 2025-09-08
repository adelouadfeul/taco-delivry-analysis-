#!/usr/bin/env python
# coding: utf-8

# #                                                    taco delivery dataset analysis

# ## 🧹 Data Cleaning Questions
#   1-Are there any missing values in the dataset?\
#   2-Are the data types appropriate (e.g., are time columns in datetime format)?\
#   3-Are there any duplicate orders (based on Order ID)?\
#   4-Are there outliers in Delivery Duration (min), Price ($), Tip ($), or Distance (km)?\
#   5-Are all values in Taco Size and Taco Type consistent (e.g., no typos or casing issues)?

# ### louading data

# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
plt.style.use('fivethirtyeight')


# In[2]:


df=pd.read_csv(r"C:\Users\KARIM\Downloads\taco_sales_(2024-2025).csv")


# In[3]:


df.head()


# # 1

# In[4]:


null_val=df.isnull().sum().sum()
if null_val ==0 :
    print('there is no missing values in the data') 
else: print('there is',null_val,' missing values')


# # 2

# In[9]:


df.info()


# In[19]:


df['Order Time']=pd.to_datetime(df['Order Time'], format="%d-%m-%Y %H:%M")
df['Delivery Time']=pd.to_datetime(df['Delivery Time'], format="%d-%m-%Y %H:%M")


# # 3

# In[17]:


duplicated_val=df.duplicated().sum()
if null_val ==0 :
    print('there is no duplicated values in the data') 
else: print('there is',duplicated_val,' dublicated values')


# # 4

# In[27]:


df.describe()[['Delivery Duration (min)','Price ($)','Tip ($)']]


# In[35]:


fig,ax=plt.subplots(1,3)
ax[0].boxplot(df['Delivery Duration (min)'])
ax[0].set_title('Delivery Duration (min) plot')
ax[1].boxplot(df['Price ($)'])
ax[1].set_title('Price ($) plot')
ax[2].boxplot(df['Tip ($)'])
ax[2].set_title('Tip ($) plot')
plt.show()


# In[31]:


cols=['Delivery Duration (min)','Tip ($)','Price ($)']
for col in cols :
    mean = df[col].mean()
    std = df[col].std()
    outliers = (df[col] < mean - 2*std) | (df[col] > mean + 2*std)
    print('there is ',outliers.sum(),' outlier in ',col,'variable')
        


# # 5

# In[104]:


print(df['Taco Size'].unique())
print(df['Taco Type'].unique())


# In[ ]:





# ## 📊 Data Exploration Questions
#   6-What are the most common taco types and sizes?\
#   7-What is the average delivery duration overall? Does it differ by taco size or type?\
#   8-Which restaurants have the fastest or slowest delivery times?\
#   9-What is the distribution of Toppings Count? Do customers usually choose many or few toppings?\
#   10-How far do deliveries usually travel?
# 
# 

# # 6

# In[38]:


taco_col=['Taco Type','Taco Size']
for col in taco_col:
    print('most common ',col,'is', df[col].value_counts(ascending=False).reset_index()[col][0])


# # 7

# In[40]:


print('the average delivery duration is ',df['Delivery Duration (min)'].mean(),'min')


# In[42]:


df.groupby('Taco Type')['Delivery Duration (min)'].mean()


# In[44]:


df.groupby('Taco Size')['Delivery Duration (min)'].mean()


# # 8

# In[46]:


resturant=df.groupby('Restaurant Name')['Delivery Duration (min)'].mean().sort_values(ascending=True).reset_index()


# In[48]:


print('the fastest resturant in delivery is ',resturant['Restaurant Name'][0],'with ',round(resturant['Delivery Duration (min)'][0]),'min average delivery time')


# In[50]:


resturant_slowest=df.groupby('Restaurant Name')['Delivery Duration (min)'].mean().sort_values(ascending=False).reset_index()


# In[52]:


print('the slowesst resturant in delivery is ',resturant_slowest['Restaurant Name'][0],'with ',round(resturant_slowest['Delivery Duration (min)'][0]),'min average delivery time')


# # 9

# In[83]:


sns.histplot(df['Toppings Count'], bins=6)
plt.title('distrbution of toppings count')
plt.show()


# In[56]:


df['Toppings Count'].value_counts()


# In[58]:


df['Toppings Count'].describe()


# # 10

# In[60]:


df['Distance (km)'].describe()


# In[85]:


sns.histplot(df['Distance (km)'], bins=30, kde=True)


# ## 💲 Price & Tip Analysis Questions
#   11-What is the average price of an order?\
#   12-Is there a relationship between price and tip amount?\
#   13-Do customers tip more on weekends?\
#   14-Does the number of toppings or taco size affect the price significantly?

# # 11

# In[65]:


print('the average price of order is ',round(df['Price ($)'].mean(),2))


# # 12

# In[89]:


sns.scatterplot(data=df, x='Price ($)', y='Tip ($)')
plt.title('Tip Amount vs. Price')
plt.xlabel('Price ($)')
plt.ylabel('Tip ($)')
plt.show()


# In[85]:


print('the correlation value is ', round(df['Price ($)'].corr(df['Tip ($)']),3))


# # 13

# In[87]:


df.groupby('Weekend Order')['Tip ($)'].mean()


# In[91]:


df.groupby('Weekend Order')['Tip ($)'].mean().plot(kind='bar')


# In[91]:


from scipy.stats import ttest_ind

weekend_tips = df[df['Weekend Order'] == True]['Tip ($)']
weekday_tips = df[df['Weekend Order'] == False]['Tip ($)']

t_stat, p_val = ttest_ind(weekend_tips, weekday_tips, equal_var=False)
print(f"T-statistic: {t_stat:.2f}, p-value: {p_val:.3f}")


# # 14

# In[93]:


df.groupby('Toppings Count')['Price ($)'].mean().sort_values(ascending=False)


# In[95]:


print('the correlation value ',df['Toppings Count'].corr(df['Price ($)']))


# In[97]:


df.groupby('Taco Size')['Price ($)'].mean().sort_values(ascending=False)


# In[99]:


import statsmodels.formula.api as smf

# Make sure categorical variable is treated properly
model = smf.ols('Q("Price ($)") ~ Q("Toppings Count") + C(Q("Taco Size"))', data=df).fit()
print(model.summary())


# ## 📆 Time-Based & Weekend Questions
#   15-Are orders more frequent on weekends vs weekdays?\
#   16-At what times of day are most orders placed?\
#   17-Is delivery slower on weekends?\
#   18-Do customers order more tacos or toppings on weekends?

# # 15

# In[102]:


df['Weekend Order'].value_counts(normalize=True)*100


# # 16

# In[124]:


df['order hour']=df['Delivery Time'].dt.hour


# In[132]:


plt.hist(x=df['order hour'],bins=24)
plt.title('hours histogram')
plt.show()


# In[190]:


df['order hour'].value_counts()


# In[192]:


df['order hour'].describe()


# # 17

# In[210]:


df['delivery time in min']=df['Delivery Time']-df['Order Time']


# In[222]:


df.groupby('Weekend Order')['delivery time in min'].mean()


# # 18

# In[183]:


df[df['Weekend Order']==True]['Toppings Count']


# In[146]:


df['Weekend Order'].value_counts()


# In[176]:


from scipy.stats import ttest_ind

weekend = df[df['Weekend Order'] == True]['Toppings Count']
weekday = df[df['Weekend Order'] == False]['Toppings Count']

t_stat, p_val = ttest_ind(weekend, weekday, equal_var=False)
print(f"T-test: t={t_stat:.2f}, p={p_val:.4f}")


# In[ ]:





# In[175]:


df.columns


# In[ ]:





# In[220]:


df['delivery time in min']


# In[ ]:




