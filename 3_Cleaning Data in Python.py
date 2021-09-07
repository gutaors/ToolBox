#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1. Exploring data

# df = pd.read_csv('my_data.csv')
# 
# df.head()
# 
# df.info()
# 
# df.columns
# 
# df.describe()
# 
# df.column.value_counts()
# 
# df.column.plot('hist')

# In[2]:


import zipfile

with zipfile.ZipFile("dob_job_application_filings_subset.csv.zip") as z:
    with z.open("dob_job_application_filings_subset.csv") as f:
# Read the file into a DataFrame: df
        df = pd.read_csv(f, low_memory=False)

# Print the shape of df
print(df.shape)

# Print the columns of df
print(df.columns)


# In[3]:


# Print the info of df
print(df.info())


# In[4]:


# Print the value counts for 'Borough'
print(df['Borough'].value_counts(dropna=False))


# In[5]:


# Describe the column
df['Existing Zoning Sqft'].describe()


# In[6]:


# Plot the histogram
df['Existing Zoning Sqft'].plot(kind='hist', rot=70, logx=True, logy=True)

# Display the histogram
plt.show()


# ## 2. Tidying data for analysis

# In[7]:


airquality = pd.read_csv('airquality.csv')
airquality.head()


# In[8]:


airquality_melt = pd.melt(airquality, id_vars=['Month', 'Day'], var_name='measurement', value_name='reading')
airquality_melt.head()


# In[9]:


airquality_pivot = airquality_melt.pivot_table(index=['Month','Day'], columns='measurement', values='reading')
airquality_pivot.head()


# In[10]:


airquality_pivot = airquality_melt.pivot_table(index=['Month', 'Day'], columns='measurement', values='reading', aggfunc=np.mean)

# Reset the index of airquality_pivot
airquality_pivot = airquality_pivot.reset_index()

# Print the head of airquality_pivot
print(airquality_pivot.head())


# In[11]:


tb = pd.read_csv('tb.csv')
# Melt tb: tb_melt
tb_melt = pd.melt(frame = tb, id_vars=['country','year'])

# Create the 'gender' column
tb_melt['gender'] = tb_melt.variable.str[0]

# Create the 'age_group' column
tb_melt['age_group'] = tb_melt.variable.str[1:]

# Print the head of tb_melt
print(tb_melt.head())


# In[12]:


ebola = pd.read_csv('ebola.csv')


# In[13]:


# Melt ebola: ebola_melt
ebola_melt = pd.melt(ebola, id_vars=['Date', 'Day'], var_name='type_country', value_name='counts')

# Create the 'str_split' column
ebola_melt['str_split'] = ebola_melt.type_country.str.split('_')

# Create the 'type' column
ebola_melt['type'] = ebola_melt['str_split'].str.get(0)

# Create the 'country' column
ebola_melt['country'] =ebola_melt['str_split'].str.get(1)

# Print the head of ebola_melt
print(ebola_melt.head())


# ## 3. Combining data for analysis
# pd.merge(df1, df2, ...) 
# 
# pd.concat([df1, df2, df3, ...])

# In[14]:


# Write the pattern: pattern
pattern = '*.csv'

# Save all file matches: csv_files
csv_files = glob.glob(pattern)

# Print the file names
print(csv_files)

# Load the second file into a DataFrame: csv2
csv2 = pd.read_csv(csv_files[1])

# Print the head of csv2
print(csv2.head())


# In[15]:


# Create an empty list: frames
frames = []

#  Iterate over csv_files
for csv in csv_files:

    #  Read csv into a DataFrame: df
    df = pd.read_csv(csv)
    
    # Append df to frames
    frames.append(df)

# Concatenate frames into a single DataFrame: uber
uber = pd.concat(frames, sort=False)

# Print the shape of uber
print(uber.shape)

# Print the head of uber
print(uber.head())


# ## 4. Cleaning data for analysis
# def cleaning_function(row_data):
# 
#     data cleaning steps
#     
#     return ...
#     
# df.apply(cleaning_function, axis=1)
# 
# assert (df.column_data > 0).all()

# In[16]:


tips = pd.read_csv('tips.csv')
tips.sample(5)


# ### Converting data types
# df.dtypes
# 
# df['column'] = df['column'].to_numeric()
# 
# df['column'] = df['column'].astype(str)

# In[17]:


# Convert the sex column to type 'category'
tips.sex = tips.sex.astype('category')

# Convert the smoker column to type 'category'
tips.smoker = tips.smoker.astype('category')

# Print the info of tips
print(tips.info())


# In[18]:


# Convert 'total_bill' to a numeric dtype
tips['total_bill'] = pd.to_numeric(tips['total_bill'], errors='coerce')

# Convert 'tip' to a numeric dtype
tips['tip'] = pd.to_numeric(tips['tip'], errors = 'coerce')

# Print the info of tips
print(tips.info())


# ### apply

# In[19]:


# Define recode_gender()
def recode_gender(gender):

    # Return 0 if gender is 'Female'
    if gender == 'Female':
        return 0
    
    # Return 1 if gender is 'Male'    
    elif gender == 'Male':
        return 1
    
    # Return np.nan    
    else:
        return np.nan

# Apply the function to the sex column
tips['recode'] = tips['sex'].apply(recode_gender)

# Print the first five rows of tips
tips.head()


# #### Write the lambda function using replace
# tips['total_dollar_replace'] = tips.total_dollar.apply(lambda x: x.replace('$', ''))
# 
# #### Write the lambda function using regular expressions
# tips['total_dollar_re'] = tips.total_dollar.apply(lambda x: re.findall('\d+\.\d+', x)[0])

# ### re

# In[20]:


# Import the regular expression module
import re

# Compile the pattern: prog
prog = re.compile('\d{3}-\d{3}-\d{4}')

# See if the pattern matches
result = prog.match('123-456-7890')
print(bool(result))

# See if the pattern matches
result2 = prog.match('1123-456-7890')
print(bool(result2))


# In[21]:


# Import the regular expression module
import re

# Find the numeric values: matches
matches = re.findall('\d+', 'the recipe calls for 10 strawberries and 1 banana')

# Print the matches
print(matches)


# In[22]:


# Write the first pattern
pattern1 = bool(re.match(pattern='\d{3}-\d{3}-\d{4}', string='123-456-7890'))
print(pattern1)

# Write the second pattern
pattern2 = bool(re.match(pattern='\$\d*\.\d{2}', string='$123.45'))
print(pattern2)

# Write the third pattern
pattern3 = bool(re.match(pattern='[A-Z]\w*', string='Australia'))
print(pattern3)


# ### Dropping duplicate data

# #### Create the new DataFrame: tracks
# tracks = billboard[['year', 'artist', 'track', 'time']]
# 
# #### Drop the duplicates: tracks_no_duplicates
# tracks_no_duplicates = tracks.drop_duplicates()

# ### Filling missing data

# In[23]:


# Calculate the mean of the Ozone column: oz_mean
oz_mean = airquality['Ozone'].mean()

# Replace all the missing values in the Ozone column with the mean
airquality['Ozone'] = airquality['Ozone'].fillna(oz_mean)

# Print the info of airquality
print(airquality.info())


# ### Testing your data with asserts

# #### Assert that there are no missing values
# assert ebola.notnull().all().all()
# 
# #### Assert that all values are >= 0
# assert (ebola >= 0).all().all()

# ## 5. Case Study

# In[24]:


gapminder = pd.read_csv('gapminder.csv')
gapminder.drop('Unnamed: 0', axis = 1, inplace = True)
gapminder.head()


# In[25]:


# Melt gapminder: gapminder_melt
gapminder_melt = pd.melt(gapminder, id_vars = 'Life expectancy')

# # Rename the columns
gapminder_melt.columns = ['country', 'year', 'life_expectancy']

# Print the head of gapminder_melt
print(gapminder_melt.head())


# In[26]:


# Convert the year column to numeric
gapminder_melt.year = pd.to_numeric(gapminder_melt.year)

# Test if country is of type object
assert gapminder_melt.country.dtypes == np.object

# Test if year is of type int64
assert gapminder_melt.year.dtypes == np.int64

# Test if life_expectancy is of type float64
assert gapminder_melt.life_expectancy.dtypes == np.float64


# In[27]:


# Create the series of countries: countries
countries = gapminder_melt.country

# Drop all the duplicates from countries
countries = countries.drop_duplicates()

# Write the regular expression: pattern
pattern = '^[A-Za-z\.\s]*$'

# Create the Boolean vector: mask
mask = countries.str.contains(pattern)

# Invert the mask: mask_inverse
mask_inverse = ~mask

# Subset countries using mask_inverse: invalid_countries
invalid_countries = countries.loc[mask_inverse]

# Print invalid_countries
print(invalid_countries)


# In[28]:


# Assert that country does not contain any missing values
assert pd.notnull(gapminder_melt.country).all()

# Assert that year does not contain any missing values
assert pd.notnull(gapminder_melt.year).all().all()

# Drop the missing values
gapminder_melt = gapminder_melt.dropna()

# Print the shape of gapminder
print(gapminder_melt.shape)


# In[29]:


# Add first subplot
plt.subplot(2, 1, 1) 

# Create a histogram of life_expectancy
gapminder_melt.life_expectancy.plot(kind = 'hist')

# Group gapminder: gapminder_agg
gapminder_agg = gapminder_melt.groupby('year')['life_expectancy'].mean()

# Print the head of gapminder_agg
print(gapminder_agg.head())

# Print the tail of gapminder_agg
print(gapminder_agg.tail())

# Add second subplot
plt.subplot(2, 1, 2)

# Create a line plot of life expectancy per year
gapminder_agg.plot()

# Add title and specify axis labels
plt.title('Life expectancy over the years')
plt.ylabel('Life expectancy')
plt.xlabel('Year')

# Display the plots
plt.tight_layout()
plt.show()

# Save both DataFrames to csv files
gapminder_melt.to_csv('gapminder_melt.csv')
gapminder_agg.to_csv('gapminder_agg.csv')


# In[ ]:




