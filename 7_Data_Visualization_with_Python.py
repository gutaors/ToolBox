#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1. Customizing plots

# In[2]:


df = pd.read_csv('percent-bachelors-degrees-women-usa.csv')
df.head()


# ### Multiple plots on single axis

# In[3]:


year = df['Year']
physical_sciences = df['Physical Sciences']
computer_science = df['Computer Science']
health = df['Health Professions']
education = df['Education']

# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')

# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science, color = 'red')


# ### Using axes()

# In[4]:


# Create plot axes for the first line plot
plt.axes([0.05, 0.05, 0.425, 0.9])

# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')

# Create plot axes for the second line plot
plt.axes([0.525, 0.05, 0.425, 0.9])

# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science, color = 'red')


# ### Using subplot() (1)

# In[5]:


# Create a figure with 1x2 subplot and make the left subplot active
plt.subplot(1,2,1)

# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')
plt.title('Physical Sciences')

# Make the right subplot active in the current 1x2 subplot grid
plt.subplot(1,2,2)

# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science, color='red')
plt.title('Computer Science')

# Use plt.tight_layout() to improve the spacing between subplots
plt.tight_layout()


# ### Using subplot() (2)

# In[6]:


# Create a figure with 2x2 subplot layout and make the top left subplot active
plt.subplot(2,2,1)

# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')
plt.title('Physical Sciences')


# Make the top right subplot active in the current 2x2 subplot grid 
plt.subplot(2,2,2)

# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science, color = 'red')
plt.title('Computer Science')

# Make the bottom left subplot active in the current 2x2 subplot grid
plt.subplot(2,2,3)

# Plot in green the % of degrees awarded to women in Health Professions
plt.plot(year, health, color='green')
plt.title('Health Professions')

# Make the bottom right subplot active in the current 2x2 subplot grid
plt.subplot(2,2,4)

# Plot in yellow the % of degrees awarded to women in Education
plt.plot(year, education, color='yellow')
plt.title('Education')

# Improve the spacing between subplots and display them
plt.tight_layout()


# ### Using xlim(), ylim()

# In[7]:


# Plot the % of degrees awarded to women in Computer Science and the Physical Sciences
plt.plot(year,computer_science, color='red') 
plt.plot(year, physical_sciences, color='blue')

# Add the axis labels
plt.xlabel('Year')
plt.ylabel('Degrees awarded to women (%)')

# Set the x-axis range
plt.xlim(1990, 2010)

# Set the y-axis range
plt.ylim(0, 50)

# Add a title and display the plot
plt.title('Degrees awarded to women (1990-2010)\nComputer Science (red)\nPhysical Sciences (blue)')
plt.show()

# Save the image as 'xlim_and_ylim.png'
plt.savefig('xlim_and_ylim.png')


# ### Using axis()

# In[8]:


# Plot in blue the % of degrees awarded to women in Computer Science
plt.plot(year,computer_science, color='blue')

# Plot in red the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences,color='red')

# Set the x-axis and y-axis limits
plt.axis((1990,2010,0,50))

# Show the figure
plt.show()

# Save the figure as 'axis_limits.png'
plt.savefig('axis_limits.png')


# ### Using legend()

# In[9]:


# Specify the label 'Computer Science'
plt.plot(year, computer_science, color='red', label='Computer Science') 

# Specify the label 'Physical Sciences' 
plt.plot(year, physical_sciences, color='blue', label = 'Physical Sciences')

# Add a legend at the lower center
plt.legend(loc = 'lower center')

# Add axis labels and title
plt.xlabel('Year')
plt.ylabel('Enrollment (%)')
plt.title('Undergraduate enrollment of women')


# ### Using annotate()

# In[10]:


# Plot with legend as before
plt.plot(year, computer_science, color='red', label='Computer Science') 
plt.plot(year, physical_sciences, color='blue', label='Physical Sciences')
plt.legend(loc='lower right')

# Compute the maximum enrollment of women in Computer Science: cs_max
cs_max = computer_science.max()

# Calculate the year in which there was maximum enrollment of women in Computer Science: yr_max
yr_max = year[computer_science.idxmax()]

# Add a black arrow annotation
plt.annotate('Maximum', xy=(yr_max, cs_max), xytext=(yr_max+5, cs_max+5), arrowprops=dict(facecolor='black'))

# Add axis labels and title
plt.xlabel('Year')
plt.ylabel('Enrollment (%)')
plt.title('Undergraduate enrollment of women')


# ### Modifying styles

# In[11]:


# Set the style to 'ggplot'
#plt.style.use('ggplot')

# Create a figure with 2x2 subplot layout
plt.subplot(2, 2, 1) 

# Plot the enrollment % of women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')
plt.title('Physical Sciences')

# Plot the enrollment % of women in Computer Science
plt.subplot(2, 2, 2)
plt.plot(year, computer_science, color='red')
plt.title('Computer Science')

# Add annotation
cs_max = computer_science.max()
yr_max = year[computer_science.idxmax()]
plt.annotate('Maximum', xy=(yr_max, cs_max), xytext=(yr_max-1, cs_max-10), arrowprops=dict(facecolor='black'))

# Plot the enrollmment % of women in Health professions
plt.subplot(2, 2, 3)
plt.plot(year, health, color='green')
plt.title('Health Professions')

# Plot the enrollment % of women in Education
plt.subplot(2, 2, 4)
plt.plot(year, education, color='yellow')
plt.title('Education')

# Improve spacing between subplots and display them
plt.tight_layout()


# ## 2. Plotting 2D arrays

# ### Generating meshes

# In[12]:


# Generate two 1-D arrays: u, v
u = np.linspace(-2, 2, 41)
v = np.linspace(-1,1,21)

# Generate 2-D arrays from u and v: X, Y
X,Y = np.meshgrid(u,v)

# Compute Z based on X and Y
Z = np.sin(3*np.sqrt(X**2 + Y**2)) 

# Display the resulting image with pcolor()
plt.pcolor(Z)
plt.show()

# Save the figure to 'sine_mesh.png'
plt.savefig('sine_mesh.png')


# In[13]:


# Generate a default contour map of the array Z
plt.subplot(2,2,1)
plt.contour(X, Y, Z)
# Generate a contour map with 20 contours
plt.subplot(2,2,2)
plt.contour(X, Y, Z,20)

# Generate a default filled contour map of the array Z
plt.subplot(2,2,3)
plt.contourf(X, Y, Z)
# Generate a default filled contour map with 20 contours
plt.subplot(2,2,4)
plt.contourf(X, Y, Z,20)

# Improve the spacing between subplots
plt.tight_layout()

# Display the figure
plt.show()


# ### Modifying colormaps

# In[14]:


# Create a filled contour plot with a color map of 'viridis'
plt.subplot(2,2,1)
plt.contourf(X,Y,Z,20, cmap='viridis')
plt.colorbar()
plt.title('Viridis')

# Create a filled contour plot with a color map of 'gray'
plt.subplot(2,2,2)
plt.contourf(X,Y,Z,20, cmap='gray')
plt.colorbar()
plt.title('Gray')

# Create a filled contour plot with a color map of 'autumn'
plt.subplot(2,2,3)
plt.contourf(X, Y, Z, 20, cmap = 'autumn')
plt.colorbar()
plt.title('Autumn')

# Create a filled contour plot with a color map of 'winter'
plt.subplot(2,2,4)
plt.contourf(X, Y, Z, 20, cmap = 'winter')
plt.colorbar()
plt.title('Winter')

# Improve the spacing between subplots and display them
plt.tight_layout()
plt.show()


# ### Using hist2d()

# In[15]:


auto = pd.read_csv('auto-mpg.csv')
hp = auto.hp
mpg = auto.mpg


# In[16]:


# Generate a 2-D histogram
plt.hist2d(hp, mpg, bins = (20,20), range = ((40,235), (8,48)))

# Add a color bar to the histogram
plt.colorbar()

# Add labels, title, and display the plot
plt.xlabel('Horse power [hp]')
plt.ylabel('Miles per gallon [mpg]')
plt.title('hist2d() plot')


# ### Using hexbin()

# In[17]:


# Generate a 2d histogram with hexagonal bins
plt.hexbin(hp, mpg, gridsize = (15, 12), extent = (40, 235, 8, 48))

           
# Add a color bar to the histogram
plt.colorbar()

# Add labels, title, and display the plot
plt.xlabel('Horse power [hp]')
plt.ylabel('Miles per gallon [mpg]')
plt.title('hexbin() plot')


# ### Loading, examining images

# In[18]:


# Load the image into an array: img
img = plt.imread('thiebaud-cakes.jpg')

# Print the shape of the image
print(img.shape)

# Display the image
plt.imshow(img)

# Hide the axes
plt.axis('off')


# ### Pseudocolor plot from image data

# In[19]:


# Load the image into an array: img
img = plt.imread('thiebaud-cakes.jpg')

# Print the shape of the image
print(img.shape)

# Compute the sum of the red, green and blue channels: intensity
intensity = img.sum(axis = 2)

# Print the shape of the intensity
print(intensity.shape)

# Display the intensity with a colormap of 'gray'
plt.imshow(intensity, cmap = 'gray')

# Add a colorbar
plt.colorbar()

# Hide the axes and show the figure
plt.axis('off')


# ### Extent and aspect

# In[20]:


# Load the image into an array: img
img = plt.imread('thiebaud-cakes.jpg')

# Specify the extent and aspect ratio of the top left subplot
plt.subplot(2,2,1)
plt.title('extent=(-1,1,-1,1),\naspect=0.5') 
plt.xticks([-1,0,1])
plt.yticks([-1,0,1])
plt.imshow(img, extent=(-1,1,-1,1), aspect =0.5)

# Specify the extent and aspect ratio of the top right subplot
plt.subplot(2,2,2)
plt.title('extent=(-1,1,-1,1),\naspect=1')
plt.xticks([-1,0,1])
plt.yticks([-1,0,1])
plt.imshow(img, extent=(-1,1,-1,1), aspect=1)

# Specify the extent and aspect ratio of the bottom left subplot
plt.subplot(2,2,3)
plt.title('extent=(-1,1,-1,1),\naspect=2')
plt.xticks([-1,0,1])
plt.yticks([-1,0,1])
plt.imshow(img, extent=(-1,1,-1,1), aspect=2)

# Specify the extent and aspect ratio of the bottom right subplot
plt.subplot(2,2,4)
plt.title('extent=(-2,2,-1,1),\naspect=2')
plt.xticks([-2,-1,0,1,2])
plt.yticks([-1,0,1])
plt.imshow(img, extent =(-2,2,-1,1), aspect = 2)

# Improve spacing and display the figure
plt.tight_layout()
plt.show()


# ### Rescaling pixel intensities

# In[21]:


# Load the image into an array: image
image = plt.imread('josef-koudelka.jpg')

# Extract minimum and maximum values from the image: pmin, pmax
pmin, pmax = image.min(), image.max()
print("The smallest & largest pixel intensities are %d & %d." % (pmin, pmax))

# Rescale the pixels: rescaled_image
rescaled_image = 256*(image - pmin) / (pmax - pmin)
print("The rescaled smallest & largest pixel intensities are %.1f & %.1f." % 
      (rescaled_image.min(), rescaled_image.max()))

# Display the original image in the top subplot
plt.subplot(2,1,1)
plt.title('original image')
plt.axis('off')
plt.imshow(image, cmap = 'gray')

# Display the rescaled image in the bottom subplot
plt.subplot(2,1,2)
plt.title('rescaled image')
plt.axis('off')
plt.imshow(rescaled_image, cmap = 'gray')


# ## 3. Statistical plots with Seaborn

# ### Simple linear regressions

# In[22]:


# Plot a linear regression between 'weight' and 'hp'
sns.lmplot(x = 'weight', y='hp', data =auto)


# ### Plotting residuals of a regression

# In[23]:


# Generate a green residual plot of the regression between 'hp' and 'mpg'
sns.residplot(x='hp', y='mpg', data=auto, color='green')


# ### Higher-order regressions

# In[24]:


# Generate a scatter plot of 'weight' and 'mpg' using red circles
plt.scatter(auto['weight'], auto['mpg'], label='data', color='red', marker='o')

# Plot in blue a linear regression of order 1 between 'weight' and 'mpg'
sns.regplot(x='weight', y='mpg', data=auto, scatter= None, color='blue', label='order 1')

# Plot in green a linear regression of order 2 between 'weight' and 'mpg'
sns.regplot(x = 'weight', y = 'mpg', data = auto, scatter = None, color = 'green', label = 'order 2', order = 2)

# Add a legend and display the plot
plt.legend(loc = 'upper right')


# ### Grouping linear regressions by hue

# In[25]:


# Plot a linear regression between 'weight' and 'hp', with a hue of 'origin' and palette of 'Set1'
sns.lmplot(x = 'weight', y = 'hp', data = auto, hue = 'origin', palette = 'Set1')


# ### Grouping linear regressions by row or column

# In[26]:


# Plot linear regressions between 'weight' and 'hp' grouped row-wise by 'origin'
sns.lmplot(x = 'weight', y = 'hp', data = auto, row = 'origin')


# ### Constructing strip plots

# In[27]:


# Make a strip plot of 'hp' grouped by 'cyl'
plt.subplot(2,1,1)
sns.stripplot(x='cyl', y='hp', data=auto)

# Make the strip plot again using jitter and a smaller point size
plt.subplot(2,1,2)
sns.stripplot(x='cyl', y='hp', data=auto, jitter = True, size = 3)


# ### Constructing swarm plots

# In[28]:


# Generate a swarm plot of 'hp' grouped horizontally by 'cyl'  
plt.subplot(2,1,1)
sns.swarmplot(x = 'cyl', y = 'hp', data = auto)

# Generate a swarm plot of 'hp' grouped vertically by 'cyl' with a hue of 'origin'
plt.subplot(2,1,2)
sns.swarmplot(x = 'hp', y = 'cyl', data = auto, hue = 'origin', orient = 'h')


# ### Constructing violin plots

# In[29]:


# Generate a violin plot of 'hp' grouped horizontally by 'cyl'
plt.subplot(2,1,1)
sns.violinplot(x='cyl', y='hp', data=auto)

# Generate the same violin plot again with a color of 'lightgray' and without inner annotations
plt.subplot(2,1,2)
sns.violinplot(x='cyl', y='hp', data=auto, color = 'lightgray', inner = None)

# Overlay a strip plot on the violin plot
sns.stripplot(x = 'cyl', y = 'hp', data = auto, jitter = True, size = 1.5)


# ### Plotting joint distributions (1)

# In[30]:


# Generate a joint plot of 'hp' and 'mpg'
sns.jointplot(x = 'hp', y = 'mpg', data = auto)


# ### Plotting joint distributions (2)

# kind='scatter' uses a scatter plot of the data points
# 
# kind='reg' uses a regression plot (default order 1)
# 
# kind='resid' uses a residual plot
# 
# kind='kde' uses a kernel density estimate of the joint distribution
# 
# kind='hex' uses a hexbin plot of the joint distribution

# In[31]:


# Generate a joint plot of 'hp' and 'mpg' using a hexbin plot
sns.jointplot(x = 'hp', y = 'mpg', data = auto, kind = 'hex')


# ### Plotting distributions pairwise (1)

# In[32]:


auto_short = auto[['mpg', 'hp', 'origin']]
# Plot the pairwise joint distributions from the DataFrame 
sns.pairplot(auto_short)


# ### Plotting distributions pairwise (2)

# kind='scatter' uses a scatter plot of the data points
# 
# kind='reg' uses a regression plot (default order 1)
# 
# kind='resid' uses a residual plot
# 
# kind='kde' uses a kernel density estimate of the joint distribution
# 
# kind='hex' uses a hexbin plot of the joint distribution

# In[33]:


# Plot the pairwise joint distributions grouped by 'origin' along with regression lines
sns.pairplot(auto_short, kind = 'reg', hue = 'origin')


# ### Visualizing correlations with a heatmap

# In[34]:


auto_short = auto.iloc[:, :5]
corr_matrix = auto_short.corr()
corr_matrix 


# In[35]:


# Visualize the covariance matrix using a heatmap
sns.heatmap(corr_matrix)


# ## 4. Analyzing time series and images

# In[36]:


stocks = pd.read_csv('stocks.csv', index_col = 'Date', parse_dates = True)
aapl = stocks['AAPL']
ibm = stocks['IBM']
csco = stocks['CSCO']
msft = stocks['MSFT']


# ### Multiple time series on common axes

# In[37]:


from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
# Plot the aapl time series in blue
plt.plot(aapl, color='blue', label='AAPL')

# Plot the ibm time series in green
plt.plot(ibm, color='green', label='IBM')

# Plot the csco time series in red
plt.plot(csco, color='red', label='CSCO')

# Plot the msft time series in magenta
plt.plot(msft, color='magenta', label='MSFT')

# Add a legend in the top left corner of the plot
plt.legend(loc='upper left')

# Specify the orientation of the xticks
plt.xticks(rotation=60)


# ### Multiple time series slices (1)

# In[38]:


# Plot the series in the top subplot in blue
plt.subplot(2,1,1)
plt.xticks(rotation=45)
plt.title('AAPL: 2001 to 2011')
plt.plot(aapl, color='blue')

# Slice aapl from '2007' to '2008' inclusive: view
view = aapl.loc['2007':'2008']

# Plot the sliced data in the bottom subplot in black
plt.subplot(2,1,2)
plt.xticks(rotation=45)
plt.title('AAPL: 2007 to 2008')
plt.plot(view, color='black')
plt.tight_layout()


# ### Multiple time series slices (2)

# In[39]:


# Slice aapl from Nov. 2007 to Apr. 2008 inclusive: view
view = aapl['2007-11':'2008-04']

# Plot the sliced series in the top subplot in red
plt.subplot(2,1,1)
plt.plot(view, color = 'red')
plt.title('AAPL: Nov. 2007 to Apr. 2008')
plt.xticks(rotation = 45)

# Reassign the series by slicing the month January 2008
view = aapl['2008-01']

# Plot the sliced series in the bottom subplot in green
plt.subplot(2,1,2)
plt.plot(view, color = 'green')
plt.title('AAPL: Jan. 2008')
plt.xticks(rotation = 45)

# Improve spacing and display the plot
plt.tight_layout()


# ### Plotting an inset view

# In[40]:


# Slice aapl from Nov. 2007 to Apr. 2008 inclusive: view
view = aapl['2007-11':'2008-04']

# Plot the entire series 
plt.plot(aapl)
plt.xticks(rotation=45)
plt.title('AAPL: 2001-2011')

# Specify the axes
plt.axes([0.25,0.5, 0.35, 0.35])

# Plot the sliced series in red using the current axes
plt.plot(view, color = 'red')
plt.xticks(rotation=45)
plt.title('2007/11-2008/04')


# ### Plotting moving averages

# In[41]:


mean_30 = aapl.rolling(window = 30).mean()
mean_75 = aapl.rolling(window = 75).mean()
mean_125 = aapl.rolling(window = 125).mean()
mean_250 = aapl.rolling(window = 250).mean()


# In[42]:


# Plot the 30-day moving average in the top left subplot in green
plt.subplot(2,2,1)
plt.plot(mean_30, color = 'green')
plt.plot(aapl, 'k-.')
plt.xticks(rotation=60)
plt.title('30d averages')

# Plot the 75-day moving average in the top right subplot in red
plt.subplot(2,2,2)
plt.plot(mean_75, 'red')
plt.plot(aapl, 'k-.')
plt.xticks(rotation=60)
plt.title('75d averages')

# Plot the 125-day moving average in the bottom left subplot in magenta
plt.subplot(2, 2, 3)
plt.plot(mean_125, color = 'magenta')
plt.plot(aapl, 'k-.')
plt.xticks(rotation=60)
plt.title('125d averages')

# Plot the 250-day moving average in the bottom right subplot in cyan
plt.subplot(2,2,4)
plt.plot(mean_250, color = 'cyan')
plt.plot(aapl, 'k-.')
plt.xticks(rotation=60)
plt.title('250d averages')


# ### Plotting moving standard deviations

# In[43]:


std_30 = aapl.rolling(window = 30).std()
std_75 = aapl.rolling(window = 75).std()
std_125 = aapl.rolling(window = 125).std()
std_250 = aapl.rolling(window = 250).std()


# In[44]:


# Plot std_30 in red
plt.plot(std_30, color = 'red', label='30d')

# Plot std_75 in cyan
plt.plot(std_75, color = 'cyan', label='75d')

# Plot std_125 in green
plt.plot(std_125, color = 'green', label ='std_125')

# Plot std_250 in magenta
plt.plot(std_250, color = 'magenta', label = 'std_250')

# Add a legend to the upper left
plt.legend(loc='upper left')

# Add a title
plt.title('Moving standard deviations')


# ### Extracting a histogram from a grayscale image

# In[45]:


# Load the image into an array: image
image = plt.imread('thiebaud-cakes.jpg')

# Display image in top subplot using color map 'gray'
plt.subplot(2,1,1)
plt.title('Original image')
plt.axis('off')
plt.imshow(image, cmap = 'Blues')

# Flatten the image into 1 dimension: pixels
pixels = image.flatten()

# Display a histogram of the pixels in the bottom subplot
plt.subplot(2,1,2)
plt.xlim((0,255))
plt.title('Normalized histogram')
plt.hist(pixels, bins = 64, range = (0,256), density = True, color = 'red', alpha = 0.4);


# ### Cumulative Distribution Function from an image histogram

# In[46]:


# Load the image into an array: image
image = plt.imread('thiebaud-cakes.jpg')

# Display image in top subplot using color map 'gray'
plt.subplot(2,1,1)
plt.imshow(image, cmap='gray')
plt.title('Original image')
plt.axis('off')

# Flatten the image into 1 dimension: pixels
pixels = image.flatten()

# Display a histogram of the pixels in the bottom subplot
plt.subplot(2,1,2)
pdf = plt.hist(pixels, bins=64, range=(0,256), density=False,
               color='red', alpha=0.4)
plt.grid(False)

# Use plt.twinx() to overlay the CDF in the bottom subplot
plt.twinx()

# Display a cumulative histogram of the pixels
cdf = plt.hist(pixels, bins=64, range=(0,256),
               cumulative=True, density=True,
               color='blue', alpha=0.4)
               
# Specify x-axis range, hide axes, add title and display plot
plt.xlim((0,256))
plt.grid(False)
plt.title('PDF & CDF (original image)')


# ### Equalizing an image histogram

# In[48]:


# Load the image into an array: image
image = plt.imread('josef-koudelka.jpg')

# Flatten the image into 1 dimension: pixels
pixels = image.flatten()

# Generate a cumulative histogram
cdf, bins, patches = plt.hist(pixels, bins=256, range=(0,256), density=True, cumulative=True)
new_pixels = np.interp(pixels, bins[:-1], cdf*255)

# Reshape new_pixels as a 2-D array: new_image
new_image = new_pixels.reshape(image.shape)

# Display the new image with 'gray' color map
plt.subplot(2,1,1)
plt.title('Equalized image')
plt.axis('off')
plt.imshow(new_image, cmap = 'gray')

# Generate a histogram of the new pixels
plt.subplot(2,1,2)
pdf = plt.hist(new_pixels, bins=64, range=(0,256), density=False,
               color='red', alpha=0.4)
plt.grid(False)

# Use plt.twinx() to overlay the CDF in the bottom subplot
plt.twinx()
plt.xlim((0,256))
plt.grid(False)

# Add title
plt.title('PDF & CDF (equalized image)')

# Generate a cumulative histogram of the new pixels
cdf = plt.hist(new_pixels, bins=64, range=(0,256),
               cumulative=True, density=True,
               color='blue', alpha=0.4)


# ### Extracting histograms from a color image

# In[49]:


# Load the image into an array: image
image = plt.imread('thiebaud-cakes.jpg')

# Display image in top subplot
plt.subplot(2,1,1)
plt.title('Original image')
plt.axis('off')
plt.imshow(image, cmap = 'gray')

# Extract 2-D arrays of the RGB channels: red, blue, green
red, green, blue = image[:,:,0], image[:,:,1], image[:,:,2]

# Flatten the 2-D arrays of the RGB channels into 1-D
red_pixels = red.flatten()
blue_pixels = green.flatten()
green_pixels = blue.flatten()

# Overlay histograms of the pixels of each color in the bottom subplot
plt.subplot(2,1,2)
plt.title('Histograms from color image')
plt.xlim((0,256))
plt.hist(red_pixels, bins=64, density=True, color='red', alpha=0.2);
plt.hist(blue_pixels, bins=64, density=True, color='blue', alpha=0.2);
plt.hist(green_pixels, bins=64, density=True, color='green', alpha=0.2);


# ### Extracting bivariate histograms from a color image

# In[50]:


# Load the image into an array: image
image = plt.imread('thiebaud-cakes.jpg')

# Extract RGB channels and flatten into 1-D array
red, blue, green = image[:,:,0], image[:,:,1], image[:,:,2]
red_pixels = red.flatten()
blue_pixels = blue.flatten()
green_pixels = green.flatten()

# Generate a 2-D histogram of the red and green pixels
plt.subplot(2,2,1)
plt.grid(False) 
plt.xticks(rotation=60)
plt.xlabel('red')
plt.ylabel('green')
plt.hist2d(x = red_pixels, y = green_pixels, bins = (32,32));

# Generate a 2-D histogram of the green and blue pixels
plt.subplot(2,2,2)
plt.grid(False)
plt.xticks(rotation=60)
plt.xlabel('green')
plt.ylabel('blue')
plt.hist2d(x = green_pixels, y = blue_pixels, bins = (32,32));

# Generate a 2-D histogram of the blue and red pixels
plt.subplot(2,2,3)
plt.grid(False)
plt.xticks(rotation=60)
plt.xlabel('blue')
plt.ylabel('red')
plt.hist2d(x = blue_pixels, y = red_pixels, bins = (32,32));


# In[ ]:




