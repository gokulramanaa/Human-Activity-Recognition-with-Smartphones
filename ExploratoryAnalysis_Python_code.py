
# coding: utf-8

# In[1]:


# Required Python Machine learning Packages
#Statistical analysis
import pandas as pd
import numpy as np
# visualization
import seaborn as sns
#Plotting graphs
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


df1 = pd.read_csv("C:/Users/byabh/Desktop/ALT/project/train.csv")
df2 = pd.read_csv("C:/Users/byabh/Desktop/ALT/project/test.csv")


# In[3]:


df1.shape


# In[4]: jhgjhggg 


df2.shape


# In[5]:


df1.info()


# In[21]:


df1.describe()


# In[6]:


df1.isnull().sum()


# In[7]:


df2.isnull().sum()


# In[8]:


df1.head()


# In[9]:


corr = df1[df1.columns].corr()
corr


# The data has 7352 observations with 563 variables with the first few columns representing the mean and standard deviations of body accelerations in 3 spatial dimensions (X, Y, Z). The last two columns are "subject" and "Acitivity" which represent the subject that the observation is taken from and the corresponding activity respectively. Let's see what activities have been recorded in this data

# In[10]:


print('Train labels', df1['Activity'].unique(), '\nTest Labels', df2['Activity'].unique())


# We have 6 activities, 3 passive (laying, standing and sitting) and 3 active (walking, walking_downstairs, walking_upstairs) which involve walking. So, each observation in the dataset represent one of the six activities whose features are recorded in the 561 variables. Our goal would be trian a machine to predict one of the six activities given a feature set of these 561 variables.
# 
# Let's check how many observations are recorded by each subject.

# In[12]:


pd.crosstab(df1.subject, df1.Activity)


# since the data is almost evenly distributed for all the activities among all the subjects, we have picked subject 21 to compare the activities with the first three variables - mean body acceleration in 3 spatial dimensions.

# In[26]:


sub21 = df1.loc[df1['subject']==21]


# In[30]:


fig = plt.figure(figsize=(32,24))
ax1 = fig.add_subplot(221)
ax1 = sns.stripplot(x='Activity', y=sub21.iloc[:,0], data=sub21, jitter=True)
ax2 = fig.add_subplot(222)
ax2 = sns.stripplot(x='Activity', y=sub21.iloc[:,1], data=sub21, jitter=True)
plt.show()


# So, the mean body acceleration is more variable for walking activities than for passive ones especially in the X direction. 

# In[28]:


sns.clustermap(sub15.iloc[:,[0,1,2]], col_cluster=False)


# Even though we see some dark spots in the X and Z directions (possibly from the walking activities), the bulk of the map is pretty homogenous and does not help much. Perhaps other attributes like maximum or minimum acceleration might give us a better insight than the average.
# 
# Plotting maximum acceleration with activity.

# In[19]:


fig = plt.figure(figsize=(32,24))
ax1 = fig.add_subplot(221)
ax1 = sns.stripplot(x='Activity', y='tBodyAcc-max()-X', data=sub15, jitter=True)
ax2 = fig.add_subplot(222)
ax2 = sns.stripplot(x='Activity', y='tBodyAcc-max()-Y', data=sub15, jitter=True)
plt.show()


# Passive activities fall mostly below the active ones. It actually makes sense that maximum acceleration is higher during the walking activities. 

# In[20]:


sns.clustermap(sub15[['tBodyAcc-max()-X', 'tBodyAcc-max()-Y', 'tBodyAcc-max()-Z']], col_cluster=False)


# We can now see the difference in the distribution between the active and passive activities with the walkdown activity (values between 0.5 and 0.8) clearly distinct from all others especially in the X-direction. The passive activities are indistinguishable and present no clear pattern in any direction (X, Y, Z).
