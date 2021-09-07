#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1. Extracting and transforming data

# 

# In[16]:


election = pd.read_csv('pennsylvania2012_turnout.csv', index_col = 'county')


# ### Slicing rows

# In[17]:


election.head()


# In[18]:


# Slice the row labels 'Perry' to 'Potter': p_counties
# Parte as linhas de Perry até Potter: p_counties ( como é linha, a cláusula é antes da vírgula
# e as palavras do filtro separadas por :)
# veja na linha acima que o indice é a coluna county, logo ela será nossa coluna a filtrar
p_counties = election.loc['Perry':'Potter', :]

# Print the p_counties DataFrame
print(p_counties)

# Slice the row labels 'Potter' to 'Perry' in reverse order: p_counties_rev
p_counties_rev = election.loc['Potter':'Perry':-1, :]

# Print the p_counties_rev DataFrame
print(p_counties_rev)


# In[19]:


print(p_counties_rev)


# ### Slicing columns

# In[ ]:





# In[20]:


# Slice the columns from the starting column to 'Obama': left_columns
left_columns = election.loc[:, :'Obama']

# Print the output of left_columns.head()
print(left_columns.head())

# Slice the columns from 'Obama' to 'winner': middle_columns
middle_columns = election.loc[:, 'Obama':'winner']

# Print the output of middle_columns.head()
print(middle_columns.head())

# Slice the columns from 'Romney' to the end: 'right_columns'
right_columns = election.loc[:, 'Romney':]

# Print the output of right_columns.head()
print(right_columns.head())


# ### Subselecting DataFrames with lists

# In[21]:


#three_counties.head()


# In[22]:


# Create the list of row labels: rows
rows = ['Philadelphia', 'Centre', 'Fulton']

# Create the list of column labels: cols
cols = ['winner', 'Obama', 'Romney']

# Create the new DataFrame: three_counties
three_counties = election.loc[rows, cols]

# Print the three_counties DataFrame
print(three_counties)


# ### Thresholding data (dados no limiar)

# In[23]:


# cria um array com uma faixa de valores baseado em um campo do dataset
# filtra o dataframe baseado neste array e joga em um novo dataframe chamado high_turnout_df
# imprime o novo dataframe


# In[24]:


# Create the boolean array: high_turnout
high_turnout = election.turnout > 70

# Filter the election DataFrame with the high_turnout array: high_turnout_df
high_turnout_df = election[high_turnout]

# Print the high_turnout_results DataFrame
print(high_turnout_df)


# ### Filtering columns using other columns

# In[25]:


election.head()


# In[26]:


# Import numpy
import numpy as np

# Create the boolean array: too_close
too_close = election.margin < 1

# Assign np.nan to the 'winner' column where the results were too close to call
# Atribua np.nan à coluna 'winner', onde os resultados foram muito próximos para serem chamados
election.loc[too_close, 'winner'] = np.nan

# Print the output of election.info()
print(election.info())
 #veja que abaixo de romney tem a coluna winner


# ### Filtering using NaNs ( removendo NaN e removendo aqueles que tem menos que 1000 preenchidos)
# 

# In[ ]:





# In[27]:


titanic = pd.read_csv('titanic.csv')

# Select the 'age' and 'cabin' columns: df
df = titanic[['age','cabin']]

# Print the shape of df
print(df.shape)
print (titanic.shape)


# In[28]:


# Drop columns in titanic with less than 1000 non-missing values
# dropa colunas com menos de 1000 valores não ausentes (menos de 1000 preenchidos -> dropa; mais de 1000 -> mantém)
titanic.dropna(thresh=1000, axis='columns').isnull().sum()


# In[29]:


# Drop rows in df with how='any' and print the shape
print(df.dropna(how = 'any').shape)

# Drop rows in df with how='all' and print the shape
print(df.dropna(how = 'all').shape)


# ### Using apply() to transform a column (apply pega o dataframe e aplica uma função def direto a uma coluna)
# 

# In[30]:


weather = pd.read_csv('pittsburgh2013.csv')


# In[31]:


# Write a function to convert degrees Fahrenheit to degrees Celsius: to_celsius
def to_celsius(F):
    return 5/9*(F - 32)

# Apply the function over 'Mean TemperatureF' and 'Mean Dew PointF': df_celsius
df_celsius = weather[['Mean TemperatureF','Mean Dew PointF']].apply(to_celsius)

# Reassign the columns df_celsius
df_celsius.columns = ['Mean TemperatureC', 'Mean Dew PointC']

# Print the output of df_celsius.head()
print(df_celsius.head())


# ### Using .map() with a dictionary (map() pega um dicionário e usa como critério para criar nova coluna com um if baseado no dicionário: se obama preenche com a palavra blue, se Romney preenche com a palavra red)
# 

# In[32]:


# Create the dictionary: red_vs_blue
red_vs_blue = {'Obama':'blue', 'Romney':'red'}

# Use the dictionary to map the 'winner' column to the new column: election['color']
election['color'] = election['winner'].map(red_vs_blue)

# Print the output of election.head()
print(election.head())


# ### Using vectorized functions

# In[33]:


# Import zscore from scipy.stats
from scipy.stats import zscore

# Call zscore with election['turnout'] as input: turnout_zscore
turnout_zscore = zscore(election['turnout'])

# Print the type of turnout_zscore
print(type(turnout_zscore))

# Assign turnout_zscore to a new column: election['turnout_zscore']
election['turnout_zscore'] = turnout_zscore

# Print the output of election.head()
print(election.head())


# #### Select columns with all nonzeros
# df2.loc[: , df2.all( )]
# #### Select columns with any nonzeros
# df2.loc[: , df2.any( )]
# #### Select columns without NaNs
# df.loc[: , df.notnull( ).all( )]
# #### Select columns with any NaNs
# df.loc[: , df.isnull( ).any( )]
# #### Drop rows with any NaNs
# df.dropna(how='any')
# 
# #### Work with string values
# df.index.str.upper( )
# 
# df.index.map(str.lower)

# ## 2. Advanced indexing

# In[34]:


sales = pd.read_csv('sales.csv', index_col = 'month')
sales


# ### Changing index of a DataFrame

# In[35]:


# Create the list of new indexes: new_idx
new_idx = [idx.upper() for idx in sales.index]

# Assign new_idx to sales.index
sales.index = new_idx

# Print the sales DataFrame
print(sales)


# ### Changing index name labels

# In[36]:


# Assign the string 'MONTHS' to sales.index.name
sales.index.name = 'MONTHS'

# Print the sales DataFrame
print(sales)

#Assign the string 'PRODUCTS' to sales.columns.name 
sales.columns.name = 'PRODUCTS'

#Print the sales dataframe again
print(sales)


# ### Building an index, then a DataFrame

# In[37]:


# Generate the list of months: months
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

# Assign months to sales.index
sales.index = months
# Print the modified sales DataFrame
print(sales)


# In[38]:


sales['state'] = ['CA','CA', 'NY', 'NY','TX','TX']
sales['month'] = sales.index
sales.set_index('state', inplace = True)


# ### Extracting data with a MultiIndex

# In[39]:


# Print sales.loc[['CA', 'TX']]
print(sales.loc[['CA', 'TX']])

# Print sales['CA':'TX']
print(sales['CA':'TX'])


# ### Setting & sorting a MultiIndex

# In[40]:


sales.reset_index(inplace = True)
# Set the index to be the columns ['state', 'month']: sales
sales = sales.set_index(['state', 'month'])

# Sort the MultiIndex: sales
sales = sales.sort_index()

# Print the sales DataFrame
print(sales)


# ### Indexing multiple levels of a MultiIndex

# In[41]:


# Look up data for NY in month 1: NY_month1
NY_month1 = sales.loc[('NY', 'Apr'), :]

# Look up data for CA and TX in month 2: CA_TX_month2
CA_TX_month2 = sales.loc[(['CA', 'TX'], 'Jun'), :]

# Look up data for all states in month 2: all_month2
all_month2 = sales.loc[(slice(None), 'Mar'), :]

NY_month1


# ### Using .loc[ ] with nonunique indexes

# In[42]:


sales.reset_index(inplace = True)
# Set the index to the column 'state': sales
sales = sales.set_index('state')

# Print the sales DataFrame
print(sales)

# Access the data from 'NY'
print(sales.loc['NY', :])


# ## 3. Rearranging and reshaping data

# In[43]:


users = pd.read_csv('users.csv', index_col = 0)
users


# ### Pivoting a single variable

# In[44]:


# Pivot the users DataFrame: visitors_pivot
visitors_pivot = users.pivot_table(index = 'weekday', columns = 'city', values = 'visitors')

# Print the pivoted DataFrame
print(visitors_pivot)


# ### Pivoting all variables

# In[45]:


# Pivot users with signups indexed by weekday and city: signups_pivot
signups_pivot =users.pivot(index = 'weekday', columns = 'city', values = 'signups')

# Print signups_pivot
print(signups_pivot)

# Pivot users pivoted by both signups and visitors: pivot
pivot = users.pivot(index = 'weekday', columns = 'city')

# Print the pivoted DataFrame
print(pivot)


# ### Stacking & unstacking

# In[55]:


# Unstack users by 'weekday': byweekday
byweekday = users.unstack(level = 'weekday')

# Print the byweekday DataFrame
print(byweekday)


# ### Restoring the index order

# In[47]:


# Stack 'city' back into the index of bycity: newusers

newusers = users.set_index(['city', 'weekday'])
print(newusers)

# Swap the levels of the index of newusers: newusers
newusers = newusers.swaplevel(0,1)

# Print newusers and verify that the index is not sorted
print(newusers)

# Sort the index of newusers: newusers
newusers = newusers.sort_index()

# Print newusers and verify that the index is now sorted
print(newusers)

# Verify that the new DataFrame is equal to the original
print(newusers.equals(users))


# ### Adding names for readability

# In[ ]:


# Reset the index: visitors_by_city_weekday
visitors_by_city_weekday = users

# Print visitors_by_city_weekday
print(visitors_by_city_weekday)

# Melt visitors_by_city_weekday: visitors
    visitors = pd.melt(visitors_by_city_weekday, id_vars=['weekday', 'city'], value_vars = ['visitors', 'signups'], var_name = 'activity', value_name= 'count')

# Print visitors
print(visitors)


# ### Going from wide to long: obtaining key-value pairs with melt( )

# In[ ]:


# Set the new index: users_idx
users_idx = users.set_index(['city', 'weekday'])

# Print the users_idx DataFrame
print(users_idx)

# Obtain the key-value pairs: kv_pairs
kv_pairs = pd.melt(users_idx, col_level=0)

# Print the key-value pairs
print(kv_pairs)


# ### Setting up a pivot table

# In[ ]:


# Create the DataFrame with the appropriate pivot table: by_city_day
by_city_day = users.pivot_table(index = 'weekday', columns = 'city')

# Print by_city_day
print(by_city_day)


# ### Using other aggregations in pivot tables

# In[ ]:


# Use a pivot table to display the count of each column: count_by_weekday1
count_by_weekday1 = users.pivot_table(index='weekday', aggfunc='count')

# Print count_by_weekday
print(count_by_weekday1)

# Replace 'aggfunc='count'' with 'aggfunc=len': count_by_weekday2
count_by_weekday2 = users.pivot_table(index='weekday', aggfunc=len)

# Verify that the same result is obtained
print('==========================================')
print(count_by_weekday1.equals(count_by_weekday2))


# ### Using margins in pivot tables

# In[ ]:


# Create the DataFrame with the appropriate pivot table: signups_and_visitors
signups_and_visitors = users.pivot_table(index = 'weekday', aggfunc = 'sum' )

# Print signups_and_visitors
print(signups_and_visitors)

# Add in the margins: signups_and_visitors_total 
signups_and_visitors_total = users.pivot_table(index = 'weekday', aggfunc = 'sum', margins = True)

# Print signups_and_visitors_total
print(signups_and_visitors_total)


# ## 4. Grouping data

# ### Grouping by multiple columns

# In[ ]:


# Group titanic by 'pclass'
by_class = titanic.groupby('pclass')

# Aggregate 'survived' column of by_class by count
count_by_class = by_class['survived'].count()

# Print count_by_class
print(count_by_class)

# Group titanic by 'embarked' and 'pclass'
by_mult = titanic.groupby(['embarked', 'pclass'])

# Aggregate 'survived' column of by_mult by count
count_mult = by_mult['survived'].count()

# Print count_mult
print(count_mult)


# ### Computing multiple aggregates of multiple columns

# In[ ]:


# Group titanic by 'pclass': by_class
by_class = titanic.groupby('pclass')

# Select 'age' and 'fare'
by_class_sub = by_class[['age','fare']]

# Aggregate by_class_sub by 'max' and 'median': aggregated
aggregated = by_class_sub.agg(['max', 'median'])

# Print the maximum age in each class
print(aggregated.loc[:, ('age','max')])

# Print the median fare in each class
print(aggregated.loc[:, ('fare', 'median')])


# In[ ]:


aggregated


# ### Filling missing data (imputation) by group

# In[ ]:


# Create a groupby object: by_sex_class
by_sex_class = titanic.groupby(['sex', 'pclass'])

# Write a function that imputes median
def impute_median(series):
    return series.fillna(series.median())

# Impute age and assign to titanic['age']
titanic.age = titanic.age.transform(impute_median)

# Print the output of titanic.tail(10)
titanic.tail(10)


# ### Grouping and filtering with .apply()

# In[ ]:


titanic['cabin'].unique()


# In[ ]:


def c_deck_survival(gr):

    c_passengers = gr['cabin'].str.startswith('C').fillna(False)
    return gr.loc[c_passengers, 'survived'].mean()


# In[ ]:


# Create a groupby object using titanic over the 'sex' column: by_sex
by_sex = titanic.groupby('sex')

# Call by_sex.apply with the function c_deck_survival
c_surv_by_sex = by_sex.apply(c_deck_survival)

# Print the survival rates
c_surv_by_sex


# ### Filtering and grouping with .map( )

# In[ ]:


# Create the Boolean Series: under10
under10 = (titanic['age'] < 10).map({True:'under 10', False:'over 10'})

# Group by under10 and compute the survival rate
survived_mean_1 = titanic.groupby(under10)['survived'].mean()
print(survived_mean_1)

# Group by under10 and pclass and compute the survival rate
survived_mean_2 = titanic.groupby([under10, 'pclass'])['survived'].mean()
print(survived_mean_2)


# In[ ]:


under10.head()


# ### Grouping by another series

# In[ ]:


# Read life_fname into a DataFrame: life
life = pd.read_csv('gapminder_life.csv', index_col='Country')

# Read regions_fname into a DataFrame: regions
regions = pd.read_csv('gapminder_region.csv', index_col = 'Country')

# Group life by regions['region']: life_by_region
life_by_region = life.groupby(regions.region)

# Print the mean over the '2010' column of life_by_region
print(life_by_region['life'].mean())


# ### Aggregating on index levels/fields

# In[ ]:


# Read the CSV file into a DataFrame and sort the index: gapminder
gapminder = pd.read_csv('gapminder_tidy.csv', index_col = ['Year','region','Country']).sort_index()

# Group gapminder by 'Year' and 'region': by_year_region
by_year_region = gapminder.groupby(level = ['Year','region'])

# Define the function to compute spread: spread
def spread(series):
    return series.max() - series.min()

# Create the dictionary: aggregator
aggregator = {'population':'sum', 'child_mortality':'mean', 'gdp':spread}

# Aggregate by_year_region using the dictionary: aggregated
aggregated = by_year_region.agg(aggregator)

# Print the last 6 entries of aggregated 
print(aggregated.tail(6))


# ### Detecting outliers with Z-Scores

# In[ ]:


# Import zscore
from scipy.stats import zscore

# Group gapminder_2010: standardized
standardized = gapminder.groupby('region')['life','fertility'].transform(zscore)

# Construct a Boolean Series to identify outliers: outliers
outliers = (standardized['life'] < -3) | (standardized['fertility'] > 3)

# Filter gapminder_2010 by the outliers: gm_outliers
gm_outliers = gapminder.loc[outliers]

# Print gm_outliers
gm_outliers.head()


# In[ ]:


def disparity(gr):
    # Compute the spread of gr['gdp']: s
    s = gr['gdp'].max() - gr['gdp'].min()
    # Compute the z-score of gr['gdp'] as (gr['gdp']-gr['gdp'].mean())/gr['gdp'].std(): z
    z = (gr['gdp'] - gr['gdp'].mean())/gr['gdp'].std()
    # Return a DataFrame with the inputs {'z(gdp)':z, 'regional spread(gdp)':s}
    return pd.DataFrame({'z(gdp)':z , 'regional spread(gdp)':s})


# In[ ]:


# Group gapminder_2010 by 'region': regional
regional = gapminder.groupby('region')

# Apply the disparity function on regional: reg_disp
reg_disp = regional.apply(disparity)

# Print the disparity of 'United States', 'United Kingdom', and 'China'
reg_disp.head()


# ### Grouping on a function of the index

# In[ ]:


sales = pd.read_csv('sales-feb-2015.csv', index_col='Date', parse_dates=True)
sales.head()


# In[ ]:


# Create a groupby object: by_day
by_day = sales.groupby(sales.index.strftime('%a'))

# Create sum: units_sum
units_sum = by_day['Units'].sum()

# Print units_sum
print(units_sum)


# ### Grouping and filtering with .filter( )

# In[ ]:


# Group sales by 'Company': by_company
by_company = sales.groupby('Company')

# Compute the sum of the 'Units' of by_company: by_com_sum
by_com_sum = by_company['Units'].sum()
print(by_com_sum)

# Filter 'Units' where the sum is > 35: by_com_filt
by_com_filt = by_company.filter(lambda g: g['Units'].sum() > 35)
print(by_com_filt)


# ## 5. Case Study

# In[ ]:


medals = pd.read_csv('all_medalists.csv')
medals.head()


# ### Using .value_counts() for ranking

# In[ ]:


# Select the 'NOC' column of medals: country_names
country_names = medals.loc[:, 'NOC']

# Count the number of medals won by each country: medal_counts
medal_counts = country_names.value_counts()

# Print top 15 countries ranked by medals
print(medal_counts.head(15))


# ### Using .pivot_table() to count medals by type

# In[ ]:


# Construct the pivot table: counted
counted = medals.pivot_table(index = 'NOC', columns = 'Medal', values = 'Athlete', aggfunc = 'count')

# Create the new column: counted['totals']
counted['totals'] = counted.sum(axis = 'columns')

# Sort counted by the 'totals' column
counted = counted.sort_values('totals', ascending = False)

# Print the top 15 rows of counted
print(counted.head(15))


# ### Applying .drop_duplicates()

# In[ ]:


# Select columns: ev_gen
ev_gen = medals[['Event_gender', 'Gender']]

# Drop duplicate pairs: ev_gen_uniques
ev_gen_uniques = ev_gen.drop_duplicates()

# Print ev_gen_uniques
print(ev_gen_uniques)


# ### Finding possible errors with .groupby()

# In[ ]:


# Group medals by the two columns: medals_by_gender
medals_by_gender = medals.groupby(['Event_gender', 'Gender'])

# Create a DataFrame with a group count: medal_count_by_gender
medal_count_by_gender = medals_by_gender.count()

# Print medal_count_by_gender
print(medal_count_by_gender)


# ### Locating suspicious data

# In[ ]:


# Create the Boolean Series: sus
sus = (medals.Event_gender == 'W') & (medals.Gender == 'Men')

# Create a DataFrame with the suspicious row: suspect
suspect = medals.loc[sus]

# Print suspect
print(suspect)


# ### Using .nunique() to rank by distinct sports

# In[ ]:


# Group medals by 'NOC': country_grouped
country_grouped = medals.groupby('NOC')

# Compute the number of distinct sports in which each country won medals: Nsports
Nsports = country_grouped['Sport'].nunique()

# Sort the values of Nsports in descending order
Nsports = Nsports.sort_values(ascending = False)

# Print the top 15 rows of Nsports
print(Nsports.head(15))


# ### Counting USA vs. USSR Cold War Olympic Sports

# In[ ]:


# Extract all rows for which the 'Edition' is between 1952 & 1988 and for which 'NOC' is either 'USA' or 'URS'

mask = (medals.Edition >= 1952) & (medals.Edition <= 1988) & (medals.NOC.isin(['USA','URS']))

# Use during_cold_war and is_usa_urs to create the DataFrame: cold_war_medals
cold_war_medals = medals.loc[mask]

# Group cold_war_medals by 'NOC'
country_grouped = cold_war_medals.groupby('NOC')

# Create Nsports
Nsports = country_grouped['Sport'].nunique().sort_values(ascending = False)

# Print Nsports
print(Nsports)


# ### Counting USA vs. USSR Cold War Olympic Medals

# In[ ]:


# Create the pivot table: medals_won_by_country
medals_won_by_country = medals.pivot_table(index = 'Edition', columns = 'NOC', values = 'Athlete', aggfunc = 'count')
# Slice medals_won_by_country: cold_war_usa_urs_medals
cold_war_usa_urs_medals = medals_won_by_country.loc[1952:1988, ['USA', 'URS']]

# Create most_medals 
most_medals = cold_war_usa_urs_medals.idxmax(axis = 'columns')
# Print most_medals.value_counts()
print(most_medals.value_counts())


# ### Visualizing USA Medal Counts by Edition: Line Plot

# In[ ]:


# Create the DataFrame: usa
usa = medals.loc[medals.NOC == 'USA']

# Group usa by ['Edition', 'Medal'] and aggregate over 'Athlete'
usa_medals_by_year = usa.groupby(['Edition', 'Medal'])['Athlete'].count()

# Reshape usa_medals_by_year by unstacking
usa_medals_by_year = usa_medals_by_year.unstack(level = 'Medal')

# Plot the DataFrame usa_medals_by_year
usa_medals_by_year.plot()
plt.show()


# ### Visualizing USA Medal Counts by Edition: Area Plot with Ordered Medals

# In[ ]:


# Redefine 'Medal' as an ordered categorical
medals.Medal = pd.Categorical(values = medals.Medal, categories = ['Bronze', 'Silver', 'Gold'], ordered = True)

# Create the DataFrame: usa
usa = medals[medals.NOC == 'USA']

# Group usa by 'Edition', 'Medal', and 'Athlete'
usa_medals_by_year = usa.groupby(['Edition', 'Medal'])['Athlete'].count()

# Reshape usa_medals_by_year by unstacking
usa_medals_by_year = usa_medals_by_year.unstack(level='Medal')

# Create an area plot of usa_medals_by_year
usa_medals_by_year.plot.area()
plt.show()


# In[ ]:




