#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import libraries

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None

# Now we need to read in the data
df = pd.read_csv(r'C:\Users\Asus\Downloads\movies.csv')


# In[3]:


# Now let's take a look at the data

df


# In[4]:


# We need to see if we have any missing data
# Let's loop through the data and see if there is anything missing

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


# In[5]:


# Data Types for our columns

print(df.dtypes)


# In[6]:


# Are there any Outliers?

df.boxplot(column=['gross'])


# In[7]:


# Create correct Year column

df['yearcorrect'] = df['released'].astype(str).str[:4]

df


# In[10]:


df = df.sort_values(by=['gross'],inplace=False, ascending=False)


# In[9]:


pd.set_option('display.max_rows',None)


# In[66]:


#Drop any duplicates

df.drop_duplicates()


# In[ ]:


# Budget high correlation
# Company High correlation


# In[19]:


# Scatter plot with budget vs gross 

plt.scatter(x=df['budget'], y=df['gross'])
plt.title('Budget vs Gross Earnings')
plt.xlabel('Gross Earnings')
plt.ylabel('Budget for Film')
plt.show()


# In[18]:


df.head()


# In[35]:


# Plot budget vs gross using seaborn

sns.regplot(x='budget',y='gross',data=df, scatter_kws={"color": "red"}, line_kws={"color":"blue"})


# In[ ]:


# Let's start Looking at correlation


# In[39]:


df.corr(method='pearson') #pearson,kendall,spearman


# In[ ]:


# High correlation between budget and gross


# In[41]:


correlation_matrix = df.corr(method='pearson')

sns.heatmap(correlation_matrix, annot=True)

plt.title('Correlation Matrix for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')

plt.show()


# In[43]:


# Looks at Company
df.head()


# In[64]:


df_numerized = df

for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes
        
df_numerized.head()


# In[63]:


df.head()


# In[56]:


correlation_matrix = df_numerized.corr(method='pearson')

sns.heatmap(correlation_matrix, annot=True)

plt.title('Correlation Matrix for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')

plt.show()


# In[57]:


df_numerized.corr()


# In[58]:


correlation_mat = df_numerized.corr()
corr_pairs = correlation_mat.unstack()
corr_pairs


# In[59]:


sorted_pairs = corr_pairs.sort_values()
sorted_pairs


# In[60]:


high_corr = sorted_pairs[(sorted_pairs)>0.5]
high_corr 


# In[ ]:


# Votes and budget have the highest correlation to gross earnings
#Company has Low correlation

