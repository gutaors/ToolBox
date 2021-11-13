#!/usr/bin/env python
# coding: utf-8

#  # Usando chaves {} para inserir texto dentro de string

# In[4]:


a = 'teste'
b = 'variavel 2'
print('texto: {1}, {0}'.format(a, b))


# In[7]:


print('texto: {0}, {1}'.format(a, b))


# In[10]:


print(f'texto: {a}')


# In[ ]:


import pandas as pd
import numpy as np
import sklearn

## Introdução ao Preprocessamento
# # Abrindo arquivo csv com codificação desconhecida (erro utf8)

# In[2]:


import chardet
with open('../MINISTERIO/LANCES/Dados_Consolidados/LANCES_FORN_PREGAO_ABERTO.csv', 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large

lances_fornecedor_pregao_aberto  =  pd.read_csv('Dados_Consolidados/LANCES_FORN_PREGAO_ABERTO.csv', encoding=result['encoding'])
#lances_fornecedor_pregao_aberto =  media_lances_dia.query("modoDisputa==1")
#lances_fornecedor_pregao_aberto =  media_lances_dia.query("modoDisputa==3")


# # Orientação a Objetos OO

# In[ ]:





# In[ ]:





# In[ ]:





# ### Missing data - rows

# In[ ]:





# In[2]:


#o que faço primeiro? ler csv e ver head
# observe que no head já dá para ver que tem muito dado faltante NaN
volunteer = pd.read_csv('volunteer_opportunities.csv')
volunteer.head()


# In[3]:


# Vamos brincar também com os dados da Hotmart 

df = pd.read_csv('../_HOTMART/sales_data.csv',encoding='iso-8859-1',delimiter =',')
df.head()
#olha que interessante, aqui não apareceu muito NaN na inspeção visual, tem que usar alguma ferramenta


# In[4]:


df.head()


# In[5]:


# Verificando quantos valores faltantes na coluna product_category
print(df['product_category'].isnull().sum())

# Subset somente com product_category preenchido
df_subset = df[df['product_category'].notnull()]
df_subset = df_subset[df_subset['purchase_value'].notnull()]
# Print out the shape of the subset
print(df_subset.shape)

#que bom, só tinha um registro NaN nesta coluna, que parece ser a primeira linha que veio com erro,
#alguma coisa de excel


# ### Convertendo tipos de colunas

# In[6]:


# Print the head of the hits column
print(volunteer["hits"].head())

# Convert the hits column to type int
volunteer["hits"] = volunteer["hits"].astype(int)

# Look at the dtypes of the dataset
print(volunteer.dtypes)


# In[24]:


# vamos ver o head do subset que criei
print(df_subset["purchase_value"].head())

# Converte purchase value de float64 para int
df_subset["purchase_value"] = df_subset["purchase_value"].astype(int)

# olha para os tipos de dados (dtypes) do dataset
print(df_subset.dtypes)


# ### Stratified sampling

# In[8]:


### O mario filho fala que amostra estratificada é perigosa pois não sabemos que os datasets futuros 
### onde vamos usar nosso modelo estarão estratificados, por isto ele recomenda 
### usar aleatória


# In[25]:


df.isnull().sum().sort_values()


# In[26]:


df.isnull().sum().sort_values().index


# In[31]:


# a lista do resultado da célula acima traz os nomes dos campos com valores vazios
# colamos estes valores aqui para eliminar os vazios de cada campo

df = df[['purchase_id', 'is_origin_page_social_network', 'product_id',
       'affiliate_id', 'producer_id', 'buyer_id', 'purchase_date',
       'product_creation_date', 'product_category', 'product_niche',
       'purchase_value', 'affiliate_commission_percentual', 'purchase_device',
       'purchase_origin']].dropna()


# In[33]:


from sklearn.model_selection import train_test_split

# cria dados com todas as colunas exceto categoria de produto
# interessante notar que temos id de produto mas vamos modelar na categoria do produto,
# depois preciso pesquisar o motivo disto
df_X = df.drop("product_category", axis=1)

# cria um dataset dom as etiquetas das categorias 
df_y = df[["product_category"]]

# Use stratified sampling to split up the dataset according to the volunteer_y dataset
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, stratify=df_y)

# Print out the category_desc counts on the training y labels
print(y_train["product_category"].value_counts())


# ## Standardizing Data

# ### Modeling without normalizing

# In[9]:


wine = pd.read_csv('wine_types.csv')
X = wine[['Proline', 'Total phenols', 'Hue', 'Nonflavanoid phenols']]
y = wine['Type']
wine.head()


# In[10]:


from sklearn.neighbors import KNeighborsClassifier 

# Split the dataset and labels into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')

# Fit the k-nearest neighbors model to the training data
knn.fit(X_train, y_train)

# Score the model on the test data
print(knn.score(X_test, y_test))


# ### Log normalization in Python

# In[11]:


# Print out the variance of the Proline column
print(wine['Proline'].var())

# Apply the log normalization function to the Proline column
wine['Proline_log'] = np.log(wine['Proline'])

# Check the variance of the Proline column again
print(wine['Proline_log'].var())


# ### KNN on non-scaled data

# In[12]:


X = wine[['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
       'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
       'Proanthocyanins', 'Color intensity', 'Hue',
       'OD280/OD315 of diluted wines', 'Proline_log']]
y = wine['Type']

# Split the dataset and labels into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Fit the k-nearest neighbors model to the training data
knn.fit(X_train, y_train)

# Score the model on the test data
print(knn.score(X_test, y_test))


# ### Scaling data - standardizing columns

# In[13]:


# Import StandardScaler from scikit-learn
from sklearn.preprocessing import StandardScaler

# Create the scaler
ss = StandardScaler()

# Take a subset of the DataFrame you want to scale 
wine_subset = wine[['Ash', 'Alcalinity of ash', 'Magnesium']]

# Apply the scaler to the DataFrame subset
wine_subset_scaled = ss.fit_transform(wine_subset)

wine_subset_scaled = pd.DataFrame(wine_subset_scaled, columns = ['Ash', 'Alcalinity of ash', 'Magnesium'])

wine_subset_scaled.head()


# In[14]:


wine = wine.drop(['Ash', 'Alcalinity of ash', 'Magnesium'], axis = 1)
wine = pd.concat([wine, wine_subset_scaled], axis = 1)
X = wine[['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
       'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
       'Proanthocyanins', 'Color intensity', 'Hue',
       'OD280/OD315 of diluted wines', 'Proline_log']]
y = wine['Type']
wine.head()


# ### KNN on scaled data

# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y)

# Fit the k-nearest neighbors model to the training data.
knn.fit(X_train, y_train)

# Score the model on the test data.
print(knn.score(X_test, y_test))


# ## Feature Engineering

# ### Encoding categorical variables - binary

# In[16]:


hiking = pd.read_json('hiking.json')
hiking.head()


# In[17]:


from sklearn.preprocessing import LabelEncoder

# Set up the LabelEncoder object
enc = LabelEncoder()

# Apply the encoding to the "Accessible" column
hiking['Accessible_enc'] = enc.fit_transform(hiking['Accessible'])

# Compare the two columns
print(hiking[['Accessible', 'Accessible_enc']].head())


# ### Encoding categorical variables - one-hot

# In[18]:


# Transform the category_desc column
category_enc = pd.get_dummies(volunteer['category_desc'])

# Take a look at the encoded columns
category_enc.head()


# ### Engineering numerical features - taking an average

# In[19]:


running_times_5k = pd.DataFrame([['Sue', 20.1, 18.5, 19.6, 20.3, 18.3, 19.36], ['Mark', 16.5, 17.1, 16.9, 17.6, 17.3, 17.08], ['Sean', 23.5, 25.1, 25.2, 24.6, 23.9, 24.46], ['Erin', 21.7, 21.1, 20.9, 22.1, 22.2, 21.6], ['Jenny', 25.8, 27.1, 26.1, 26.7, 26.9, 26.52], ['Russell', 30.9, 29.6, 31.4, 30.4, 29.9, 30.440000000000005]])


# In[20]:


running_times_5k.columns =  ['name', 'run1', 'run2', 'run3', 'run4', 'run5', 'x']


# In[21]:


# Create a list of the columns to average
run_columns = ["run1", "run2", "run3", "run4", "run5"]

# Use apply to create a mean column
running_times_5k["mean"] = running_times_5k.apply(lambda row: row[run_columns].mean(), axis=1)

# Take a look at the results
print(running_times_5k)


# ### Engineering numerical features - datetime

# In[22]:


# First, convert string column to date column
volunteer["start_date_converted"] = pd.to_datetime(volunteer["start_date_date"])

# Extract just the month from the converted column
volunteer["start_date_month"] = volunteer["start_date_converted"].apply(lambda row: row.month)

# Take a look at the converted and new month columns
volunteer[['start_date_converted', 'start_date_month']].head()


# ### Engineering features from strings - extraction

# In[23]:


import re
# Write a pattern to extract numbers and decimals
def return_mileage(length):
    pattern = re.compile(r"\d+\.\d+")
    
    # Search the text for matches
    mile = re.match(pattern, length)
    
    # If a value is returned, use group(0) to return the found value
    if mile is not None:
        return float(mile.group(0))
        
# Apply the function to the Length column and take a look at both columns
hiking["Length_num"] = hiking["Length"].apply(lambda row: return_mileage(str(row)))
print(hiking[["Length", "Length_num"]].head())


# ### Engineering features from strings - tf/idf

# In[24]:


from sklearn.feature_extraction.text import TfidfVectorizer
# Take the title text
title_text = volunteer['title']

# Create the vectorizer method
tfidf_vec = TfidfVectorizer()

# Transform the text into tf-idf vectors
text_tfidf = tfidf_vec.fit_transform(title_text)


# ### Text classification using tf/idf vectors

# In[25]:


from sklearn.naive_bayes import GaussianNB

# Split the dataset according to the class distribution of category_desc
y = volunteer["category_desc"]
X_train, X_test, y_train, y_test = train_test_split(text_tfidf.toarray(), y, stratify = y)

# Fit the model to the training data
nb = GaussianNB(priors=None)
nb.fit(X_train, y_train)

# Print out the model's accuracy
print(nb.score(X_test, y_test))


# ## Selecting features for modeling

# ### Selecting relevant features

# In[26]:


# Create a list of redundant column names to drop
to_drop = ["category_desc", "created_date", "locality", "region", "vol_requests"]

# Drop those columns from the dataset
volunteer_subset = volunteer.drop(to_drop, axis=1)

# Print out the head of the new dataset
volunteer_subset.head()


# ### Checking for correlated features

# In[27]:


import seaborn as sns


# In[28]:


wine_corr = wine.corr()

sns.heatmap(wine_corr, cmap = 'YlGnBu')


# In[29]:


# Take a minute to find the column where the correlation value is greater than 0.75 at least twice
to_drop = "Proline"

# Drop that column from the DataFrame
wine = wine.drop(to_drop, axis=1)


# ### Exploring text vectors, part 1

# In[30]:


vocab = {1048: 'web', 278: 'designer', 1017: 'urban', 38: 'adventures', 490: 'ice', 890: 'skating', 90: 'at', 559: 'lasker', 832: 'rink', 368: 'fight', 423: 'global', 487: 'hunger', 68: 'and', 944: 'support', 1061: 'women', 356: 'farmers', 535: 'join', 969: 'the', 708: 'oxfam', 27: 'action', 240: 'corps', 498: 'in', 680: 'nyc', 922: 'stop', 947: 'swap', 790: 'queens', 911: 'staff', 281: 'development', 992: 'trainer', 200: 'claro', 145: 'brooklyn', 1037: 'volunteer', 93: 'attorney', 221: 'community', 455: 'health', 43: 'advocates', 942: 'supervise', 189: 'children', 466: 'highland', 717: 'park', 409: 'garden', 1071: 'worldofmoney', 696: 'org', 1085: 'youth', 60: 'amazing', 791: 'race', 789: 'qualified', 133: 'board', 620: 'member', 860: 'seats', 98: 'available', 1083: 'young', 33: 'adult', 1006: 'tutor', 1016: 'updated', 11: '30', 0: '11', 513: 'insurance', 199: 'claims', 600: 'manager', 979: 'timebanksnyc', 432: 'great', 340: 'exchange', 205: 'clean', 1015: 'up', 81: 'asbury', 171: 'cementary', 918: 'staten', 524: 'island', 869: 'senior', 194: 'citizen', 392: 'friendly', 1033: 'visitor', 881: 'shop', 1000: 'tree', 161: 'care', 1068: 'workshop', 4: '20', 646: 'movie', 856: 'screener', 380: 'for', 870: 'seniors', 355: 'farm', 430: 'graphic', 691: 'open', 480: 'house', 416: 'get', 984: 'tools', 980: 'to', 806: 'recycling', 1039: 'volunteers', 660: 'needed', 353: 'family', 336: 'event', 207: 'clerical', 158: 'cancer', 1041: 'walk', 120: 'befitnyc', 739: 'physical', 30: 'activity', 700: 'organizers', 269: 'decision', 266: 'day', 5: '2011', 661: 'needs', 1084: 'your', 459: 'help', 405: 'gain', 1021: 'valuable', 245: 'counseling', 344: 'experience', 687: 'on', 845: 'samaritans', 9: '24', 479: 'hour', 255: 'crisis', 478: 'hotline', 457: 'heart', 407: 'gallery', 703: 'our', 503: 'info', 949: 'table', 373: 'finding', 471: 'homes', 542: 'kids', 1077: 'yiddish', 903: 'speaking', 472: 'homework', 460: 'helper', 892: 'skilled', 800: 'rebuilding', 982: 'together', 468: 'home', 818: 'repairs', 438: 'greenteam', 40: 'advetures', 940: 'summer', 931: 'streets', 1005: 'tuesday', 335: 'evenings', 1060: 'with', 612: 'masa', 594: 'lunch', 770: 'program', 1018: 'us', 706: 'outreach', 618: 'meals', 760: 'preparedness', 222: 'compost', 773: 'project', 613: 'master', 223: 'composter', 178: 'certificate', 249: 'course', 318: 'emblemhealth', 144: 'bronx', 683: 'of', 873: 'service', 531: 'jcc', 601: 'manhattan', 418: 'girl', 855: 'scout', 872: 'series', 296: 'dorot', 838: 'rosh', 452: 'hashanah', 709: 'package', 274: 'delivery', 713: 'painting', 511: 'instructor', 530: 'jasa', 464: 'hes', 172: 'center', 12: '3rd', 70: 'annual', 377: 'flyny', 548: 'kite', 366: 'festival', 983: 'tomorrow', 151: 'business', 566: 'leaders', 955: 'teach', 110: 'basics', 465: 'high', 852: 'schoolers', 410: 'gardening', 397: 'ft', 1004: 'tryon', 910: 'st', 610: 'martin', 748: 'poetry', 668: 'new', 1079: 'york', 216: 'college', 424: 'goal', 941: 'sunday', 361: 'february', 6: '2012', 262: 'dance', 8: '22nd', 560: 'latino', 604: 'march', 2: '17', 1013: 'university', 848: 'saturday', 1008: 'tutors', 744: 'planet', 485: 'human', 602: 'mapping', 420: 'give', 1050: 'week', 186: 'child', 569: 'learn', 796: 'read', 926: 'storytelling', 243: 'costume', 597: 'making', 912: 'stage', 277: 'design', 319: 'emergency', 351: 'fair', 17: '9th', 1053: 'west', 887: 'side', 248: 'county', 676: 'nutrition', 314: 'educator', 879: 'shape', 306: 'east', 13: '54st', 801: 'rec', 1046: 'water', 45: 'aerobics', 83: 'asser', 573: 'levy', 712: 'paint', 57: 'alongside', 783: 'publicolor', 936: 'students', 536: 'jumpstart', 797: 'readers', 564: 'lead', 252: 'crafts', 408: 'games', 348: 'face', 751: 'popcorn', 527: 'jackie', 835: 'robinson', 716: 'parent', 375: 'fitness', 916: 'starrett', 197: 'city', 585: 'line', 263: 'dancer', 615: 'math', 587: 'literacy', 114: 'be', 209: 'climb', 985: 'top', 608: 'marketing', 86: 'assistant', 313: 'education', 673: 'nonprofit', 867: 'seeks', 805: 'recruitment', 626: 'mentors', 810: 'register', 92: 'attend', 142: 'breakfast', 701: 'orientation', 529: 'january', 272: 'deliver', 1058: 'winter', 1031: 'visit', 65: 'an', 525: 'isolated', 342: 'exercise', 213: 'coach', 670: 'night', 115: 'beach', 180: 'change', 77: 'art', 772: 'programs', 229: 'consumer', 779: 'protection', 562: 'law', 589: 'liver', 579: 'life', 565: 'leader', 901: 'soup', 547: 'kitchen', 307: 'eastern', 534: 'john', 650: 'muir', 930: 'street', 1024: 'vendor', 641: 'monthly', 959: 'team', 367: 'fiesta', 977: 'throgs', 658: 'neck', 224: 'computer', 956: 'teacher', 567: 'leadership', 244: 'council', 693: 'opportunity', 231: 'conversation', 461: 'helpers', 427: 'grades', 714: 'pantry', 288: 'distribution', 305: 'earth', 960: 'tech', 1049: 'website', 692: 'opportunities', 175: 'cents', 19: 'ability', 203: 'classroom', 877: 'set', 146: 'brush', 545: 'kindness', 999: 'transportation', 58: 'alternatives', 129: 'bike', 1020: 'valet', 1026: 'video', 311: 'editing', 767: 'professionals', 921: 'stipend', 49: 'after', 851: 'school', 624: 'mentor', 666: 'networking', 138: 'bowling', 398: 'fun', 449: 'harlem', 555: 'lanes', 866: 'seeking', 1078: 'yoga', 902: 'spanish', 695: 'or', 389: 'french', 362: 'feed', 488: 'hungry', 1080: 'yorkers', 14: '55', 690: 'only', 735: 'phone', 106: 'bank', 819: 'representative', 795: 'reach', 704: 'out', 643: 'morris', 458: 'heights', 904: 'special', 155: 'camp', 946: 'susan', 551: 'komen', 259: 'cure', 433: 'greater', 47: 'affiliate', 303: 'dumbo', 79: 'arts', 698: 'organizational', 148: 'budget', 639: 'money', 596: 'makes', 871: 'sense', 994: 'training', 889: 'site', 1027: 'videographer', 376: 'fly', 152: 'by', 970: 'theater', 429: 'grant', 1074: 'writer', 745: 'planning', 778: 'proposal', 759: 'preparation', 399: 'fund', 793: 'raising', 450: 'harm', 808: 'reduction', 35: 'adv', 515: 'intern', 875: 'serving', 575: 'lgbt', 34: 'adults', 482: 'how', 830: 'ride', 130: 'bikes', 821: 'research', 401: 'fundraising', 280: 'developement', 233: 'cook', 840: 'row', 50: 'afterschool', 630: 'middle', 885: 'shower', 400: 'fundraisers', 526: 'it', 519: 'interpreters', 563: 'lawyers', 446: 'haitian', 18: 'abe', 757: 'pre', 412: 'ged', 640: 'monitor', 89: 'astoria', 634: 'million', 1001: 'trees', 421: 'giveaway', 290: 'do', 1081: 'you', 1044: 'want', 595: 'make', 283: 'difference', 204: 'classwish', 896: 'snow', 883: 'shoveling', 196: 'citizenship', 761: 'press', 586: 'list', 781: 'public', 813: 'relations', 743: 'plan', 829: 'review', 394: 'friendship', 753: 'positive', 121: 'beginnings', 546: 'kit', 611: 'mary', 803: 'recreation', 291: 'does', 697: 'organization', 659: 'need', 858: 'search', 928: 'strategy', 332: 'esl', 46: 'affected', 924: 'storm', 995: 'transform', 590: 'lives', 933: 'strengthen', 220: 'communities', 119: 'become', 302: 'driver', 1025: 'veterans', 191: 'chinese', 997: 'translator', 512: 'instructors', 653: 'museum', 621: 'membership', 275: 'department', 284: 'director', 117: 'beautify', 996: 'transitional', 822: 'residence', 470: 'homeless', 623: 'men', 953: 'tank', 517: 'internship', 774: 'projects', 841: 'run', 1056: 'wild', 139: 'boys', 475: 'hope', 419: 'girls', 219: 'communications', 792: 'raise', 100: 'awareness', 31: 'administrative', 56: 'alliance', 811: 'registrar', 647: 'ms', 1062: 'word', 162: 'career', 246: 'counselor', 722: 'passover', 304: 'early', 188: 'childhood', 149: 'build', 747: 'plastic', 137: 'bottle', 857: 'sculpture', 763: 'pride', 523: 'is', 538: 'just', 76: 'around', 238: 'corner', 520: 'involved', 675: 'now', 390: 'fresh', 53: 'air', 957: 'teachers', 372: 'find', 729: 'perfect', 533: 'job', 684: 'office', 1075: 'writing', 264: 'data', 326: 'entry', 29: 'activism', 738: 'photography', 843: 'salesforce', 265: 'database', 261: 'customization', 736: 'photo', 333: 'essay', 572: 'legal', 42: 'advisor', 467: 'hike', 974: 'thon', 236: 'coordinator', 558: 'laser', 950: 'tag', 298: 'dowling', 3: '175th', 505: 'information', 962: 'technology', 352: 'fall', 382: 'forest', 826: 'restoration', 541: 'kickoff', 1002: 'trevor', 582: 'lifeline', 247: 'counselors', 973: 'thomas', 532: 'jefferson', 614: 'materials', 1076: 'year', 386: 'founder', 341: 'executive', 453: 'haunted', 557: 'lantern', 989: 'tours', 383: 'fort', 986: 'totten', 657: 'national', 878: 'sexual', 82: 'assault', 689: 'online', 993: 'trainers', 48: 'african', 63: 'american', 210: 'clothing', 301: 'drive', 828: 'returning', 865: 'seeds', 939: 'success', 746: 'plant', 981: 'today', 443: 'growth', 1009: 'udec', 328: 'enviromedia', 636: 'mobile', 606: 'maritime', 102: 'bacchanal', 742: 'pirates', 365: 'fest', 492: 'ikea', 329: 'erie', 111: 'basin', 282: 'diabetes', 88: 'association', 364: 'feria', 267: 'de', 844: 'salud', 664: 'nepali', 105: 'bangla', 784: 'punjabi', 998: 'translators', 674: 'not', 769: 'profit', 741: 'pioneer', 159: 'capoeira', 1023: 'various', 752: 'positions', 287: 'dispatcher', 991: 'trainee', 506: 'ing', 603: 'marathon', 388: 'free', 593: 'love', 135: 'books', 268: 'dear', 96: 'authors', 52: 'aide', 850: 'scheuer', 627: 'merchandise', 293: 'donate', 943: 'supplies', 360: 'feast', 406: 'gala', 112: 'battery', 833: 'rise', 919: 'stay', 787: 'put', 820: 'rescue', 897: 'soccer', 402: 'futsal', 730: 'performing', 36: 'advanced', 202: 'classes', 1070: 'world', 854: 'science', 1054: 'western', 64: 'americorps', 25: 'aces', 310: 'economic', 864: 'security', 507: 'initiative', 331: 'esi', 633: 'mill', 173: 'centers', 631: 'midtown', 1088: 'zumba', 1030: 'vision', 635: 'mission', 66: 'analysis', 552: 'lab', 958: 'teaching', 84: 'assist', 827: 'resume', 150: 'building', 899: 'society', 214: 'coaches', 1040: 'vs', 218: 'committee', 842: 'russian', 385: 'foster', 170: 'celebration', 616: 'may', 7: '21th', 688: 'one', 711: 'pager', 294: 'donation', 489: 'hurricane', 521: 'irene', 354: 'far', 836: 'rockaway', 325: 'enjoy', 1066: 'working', 686: 'olympics', 988: 'tournament', 798: 'reading', 719: 'partners', 234: 'cooper', 909: 'square', 975: 'thrift', 908: 'spring', 166: 'case', 599: 'management', 404: 'fvcp', 990: 'trail', 254: 'crew', 447: 'halloween', 165: 'carnival', 1042: 'walkathon', 359: 'feasibility', 67: 'analyst', 749: 'police', 868: 'seminar', 1064: 'work', 1035: 'visually', 496: 'impaired', 964: 'teens', 972: 'this', 322: 'energy', 315: 'efficiency', 321: 'end', 859: 'season', 156: 'campaign', 123: 'benefits', 802: 'reception', 300: 'drill', 237: 'copywriting', 235: 'coord', 454: 'have', 725: 'penchant', 55: 'all', 971: 'things', 1028: 'vintage', 976: 'thriftshop', 718: 'partner', 726: 'pencil', 720: 'partnership', 710: 'packing', 16: '8th', 907: 'sports', 346: 'expo', 164: 'cares', 184: 'cheerleaders', 1045: 'wanted', 445: 'habitat', 371: 'finance', 215: 'coffee', 324: 'english', 755: 'practice', 570: 'learners', 456: 'healthy', 28: 'active', 978: 'time', 122: 'benefit', 73: 'april', 357: 'fashion', 929: 'strawberry', 87: 'assistants', 174: 'central', 1087: 'zoo', 1: '125th', 127: 'bideawee', 440: 'greeters', 592: 'looking', 799: 'real', 495: 'impact', 504: 'inform', 728: 'people', 756: 'practices', 580: 'lifebeat', 413: 'general', 932: 'streetsquash', 286: 'discovery', 874: 'services', 663: 'neighborhood', 768: 'profiles', 951: 'take', 915: 'stand', 51: 'against', 1029: 'violence', 345: 'expert', 41: 'advice', 537: 'june', 849: 'schedule', 258: 'crowdfunding', 727: 'penny', 451: 'harvest', 434: 'green', 185: 'chefs', 677: 'nutritionists', 379: 'foodies', 625: 'mentoring', 136: 'boom', 669: 'newsletter', 217: 'come', 934: 'strides', 1043: 'walks', 187: 'childcare', 898: 'social', 619: 'media', 422: 'giving', 157: 'can', 61: 'ambassador', 10: '2nd', 967: 'thanksgiving', 363: 'feeding', 662: 'needy', 782: 'publicity', 723: 'patient', 163: 'caregiver', 1032: 'visiting', 469: 'homebound', 358: 'fc', 679: 'nyawc', 384: 'forum', 21: 'about', 1038: 'volunteering', 809: 'refreshments', 847: 'sara', 837: 'roosevelt', 206: 'cleanup', 116: 'beautification', 337: 'events', 69: 'animal', 484: 'hudson', 834: 'river', 605: 'mariners', 825: 'response', 343: 'exhibit', 20: 'aboard', 584: 'lilac', 208: 'client', 1052: 'welcome', 279: 'desk', 685: 'older', 574: 'lexington', 251: 'craft', 750: 'poll', 1065: 'workers', 518: 'interperters', 24: 'accounting', 85: 'assistance', 477: 'hosting', 776: 'promotion', 1011: 'unicef', 954: 'tap', 814: 'release', 270: 'dedication', 771: 'programming', 500: 'incarnation', 295: 'donor', 544: 'kieran', 906: 'sponsorship', 1069: 'workshops', 118: 'because', 338: 'every', 276: 'deserves', 179: 'chance', 740: 'pin', 273: 'delivered', 886: 'shred', 15: '5th', 99: 'avenue', 169: 'cdsc', 917: 'starving', 78: 'artist', 884: 'show', 948: 'system', 396: 'front', 880: 'share', 553: 'lanch', 935: 'student', 463: 'hemophilia', 577: 'liason', 629: 'methodist', 476: 'hospital', 113: 'bay', 831: 'ridge', 124: 'benonhurst', 75: 'area', 900: 'sought', 97: 'autistic', 297: 'douglaston', 788: 'qns', 812: 'registration', 32: 'administrator', 153: 'call', 426: 'governor', 804: 'recruiter', 786: 'purim', 327: 'envelope', 938: 'stuffing', 528: 'jam', 462: 'helpline', 923: 'store', 374: 'first', 415: 'generation', 1022: 'van', 241: 'cortlandt', 816: 'remembrance', 945: 'survey', 823: 'resonations', 143: 'breast', 323: 'engine', 694: 'optimization', 622: 'memorial', 894: 'sloan', 540: 'kettering', 435: 'greenhouse', 436: 'greening', 227: 'concert', 334: 'evacuation', 824: 'resources', 417: 'gift', 126: 'bicycling', 656: 'my', 393: 'friends', 473: 'honor', 1051: 'weekend', 731: 'person', 651: 'mural', 312: 'editor', 732: 'personal', 882: 'shopper', 764: 'pro', 134: 'bono', 253: 'create', 160: 'cards', 920: 'step', 672: 'non', 780: 'provider', 516: 'interns', 645: 'motion', 431: 'graphics', 125: 'best', 147: 'buddies', 502: 'inern', 103: 'back', 588: 'little', 242: 'cosmetologist', 107: 'barber', 1036: 'vocational', 72: 'apartment', 439: 'greeter', 766: 'professional', 1019: 'use', 893: 'skills', 702: 'others', 369: 'figure', 257: 'croton', 190: 'chinatown', 193: 'ci', 758: 'prep', 239: 'corporate', 1063: 'wordpress', 132: 'blog', 510: 'instructer', 807: 'red', 474: 'hook', 289: 'divert', 966: 'textiles', 395: 'from', 554: 'landfill', 437: 'greenmarket', 965: 'textile', 154: 'calling', 195: 'citizens', 497: 'improve', 26: 'achievement', 721: 'passion', 481: 'housing', 1067: 'works', 499: 'inc', 441: 'group', 299: 'drama', 561: 'laundromats', 320: 'employment', 927: 'strategic', 667: 'never', 104: 'bad', 391: 'friend', 403: 'future', 201: 'class', 1059: 'wish', 387: 'fpcj', 1072: 'worship', 1010: 'undergraduate', 428: 'graduate', 228: 'conference', 1047: 'we', 775: 'promote', 550: 'knowledge', 715: 'parade', 74: 'archivist', 425: 'google', 44: 'adwords', 493: 'imentor', 642: 'more', 598: 'male', 632: 'miles', 637: 'moms', 183: 'charity', 176: 'century', 987: 'tour', 198: 'civil', 724: 'patrol', 62: 'america', 539: 'kept', 862: 'secret', 648: 'ms131', 549: 'knitter', 256: 'crochet', 131: 'blankets', 177: 'ceo', 591: 'logo', 1012: 'unique', 1057: 'will', 128: 'big', 37: 'adventure', 23: 'accountant', 876: 'session', 888: 'single', 644: 'mothers', 192: 'choice', 895: 'smc', 1055: 'wii', 705: 'outdoor', 671: 'nights', 607: 'market', 514: 'intake', 638: 'monday', 141: 'branding', 140: 'brand', 491: 'identity', 649: 'mt', 1086: 'zion', 543: 'kidz', 817: 'reorganize', 578: 'library', 378: 'food', 91: 'athletic', 568: 'league', 655: 'musician', 59: 'alzheimer', 654: 'music', 109: 'bash', 765: 'proctor', 952: 'taking', 339: 'exams', 777: 'promotional', 733: 'personnel', 95: 'august', 891: 'skill', 665: 'networker', 309: 'ecological', 785: 'puppet', 501: 'income', 414: 'generating', 699: 'organizations', 250: 'cpr', 576: 'lgbtq', 317: 'el', 652: 'museo', 271: 'del', 108: 'barrio', 628: 'met', 330: 'escort', 846: 'sand', 167: 'castle', 230: 'contest', 853: 'schools', 486: 'humanities', 80: 'as', 861: 'second', 556: 'language', 101: 'babies', 963: 'teen', 54: 'al', 682: 'oerter', 483: 'html', 260: 'curriculum', 737: 'photographer', 863: 'secretary', 754: 'pr', 1073: 'would', 583: 'like', 225: 'computers', 961: 'technical', 442: 'grownyc', 968: 'that', 347: 'extraordinary', 381: 'foreclosure', 762: 'prevention', 681: 'nylag', 678: 'ny', 226: 'concern', 509: 'inspire', 22: 'academic', 1007: 'tutoring', 794: 'rbi', 71: 'anyone', 211: 'cma', 212: 'cms', 232: 'conversion', 308: 'eating', 571: 'learning', 181: 'chaperones', 1034: 'visits', 411: 'gear', 1014: 'unlimited', 581: 'lifeguard', 350: 'facilitators', 1003: 'troop', 839: 'route', 609: 'marshall', 508: 'inmotion', 925: 'story', 913: 'stair', 292: 'domestic', 168: 'catskills', 815: 'relief', 316: 'effort', 94: 'audience', 734: 'pharmacy', 444: 'guide', 707: 'overnight', 494: 'immediate', 285: 'dirty', 448: 'hands', 349: 'facilitator', 905: 'specialist', 182: 'chapter', 914: 'stamps', 522: 'iridescent', 937: 'studio', 39: 'advertising', 370: 'filmmakers', 617: 'mayor', 1082: 'youcantoo'}


# In[31]:


# Add in the rest of the parameters
def return_weights(vocab, original_vocab, vector, vector_index, top_n):
    zipped = dict(zip(vector[vector_index].indices, vector[vector_index].data))
    
    # Let's transform that zipped dict into a series
    zipped_series = pd.Series({vocab[i]:zipped[i] for i in vector[vector_index].indices})
    
    # Let's sort the series to pull out the top n weighted words
    zipped_index = zipped_series.sort_values(ascending=False)[:top_n].index
    return [original_vocab[i] for i in zipped_index]

# Print out the weighted words
print(return_weights(vocab, tfidf_vec.vocabulary_, text_tfidf, 8, 3))


# ### Exploring text vectors, part 2

# In[32]:


# def words_to_filter(vocab, original_vocab, vector, top_n):
#     filter_list = []
#     for i in range(0, vector.shape[0]):
    
#         # Here we'll call the function from the previous exercise, and extend the list we're creating
#         filtered = return_weights(vocab, original_vocab, vector, i, top_n)
#         filter_list.extend(filtered)
#     # Return the list in a set, so we don't get duplicate word indices
#     return set(filter_list)

# # Call the function to get the list of word indices
# filtered_words = words_to_filter(vocab, tfidf_vec.vocabulary_, text_tfidf, 3)

# # By converting filtered_words back to a list, we can use it to filter the columns in the text vector
# filtered_text = text_tfidf[:, list(filtered_words)]


# ### Training Naive Bayes with feature selection

# In[33]:


# # Split the dataset according to the class distribution of category_desc, using the filtered_text vector
# train_X, test_X, train_y, test_y = train_test_split(filtered_text.toarray(), y, stratify=y)

# # Fit the model to the training data
# nb.fit(train_X, train_y)

# # Print out the model's accuracy
# print(nb.score(test_X, test_y))


# ### Using PCA

# In[34]:


from sklearn.decomposition import PCA

# Set up PCA and the X vector for diminsionality reduction
pca = PCA()
wine_X = wine.drop("Type", axis=1)
y = wine['Type']
# Apply PCA to the wine dataset X vector
transformed_X = pca.fit_transform(wine_X)

# Look at the percentage of variance explained by the different components
print(pca.explained_variance_ratio_)


# ### Training a model with PCA

# In[35]:


# Split the transformed X and the y labels into training and test sets
X_wine_train, X_wine_test, y_wine_train, y_wine_test = train_test_split(transformed_X, y)

# Fit knn to the training data
knn.fit(X_wine_train, y_wine_train)

# Score knn on the test data and print it out
knn.score(X_wine_test, y_wine_test)


# ## Putting it all together

# ### Checking column types

# In[36]:


ufo = pd.read_csv('ufo_sightings_large.csv')
ufo.head()


# In[37]:


# Check the column types
print(ufo.dtypes)

# Change the type of seconds to float
ufo["seconds"] = ufo["seconds"].astype(float)

# Change the date column to type datetime
ufo["date"] = pd.to_datetime(ufo["date"])

# Check the column types
print(ufo[["seconds", "date"]].dtypes)


# ### Dropping missing data

# In[38]:


# Check how many values are missing in the length_of_time, state, and type columns
print(ufo[["length_of_time", "state", "type"]].isnull().sum())

# Keep only rows where length_of_time, state, and type are not null
ufo = ufo[ufo["length_of_time"].notnull() & 
          ufo["state"].notnull() & 
          ufo["type"].notnull()]

# Print out the shape of the new dataset
print(ufo.shape)


# ### Extracting numbers from strings

# In[39]:


def return_minutes(time_string):

    # Use \d+ to grab digits
    pattern = re.compile(r"\d+")

    # Use match on the pattern and column
    num = re.match(pattern, time_string)
    if num is not None:
        return int(num.group(0))
        
# Apply the extraction to the length_of_time column
ufo["minutes"] = ufo["length_of_time"].apply(lambda x: return_minutes(str(x)))

# Take a look at the head of both of the columns
ufo[['length_of_time', 'minutes']].head()


# ### Identifying features for standardization

# In[40]:


# Check the variance of the seconds and minutes columns
print(ufo[['seconds', 'minutes']].var())

# Log normalize the seconds column
ufo["seconds_log"] = np.log(ufo["seconds"])

# Print out the variance of just the seconds_log column
print(ufo["seconds_log"].var())


# ### Encoding categorical variables

# In[41]:


# Use Pandas to encode us values as 1 and others as 0
ufo["country_enc"] = ufo["country"].apply(lambda x: 1 if x == 'us'else 0)

# Print the number of unique type values
print(len(ufo.type.unique()))

# Create a one-hot encoded set of the type values
type_set = pd.get_dummies(ufo['type'])

# Concatenate this set back to the ufo DataFrame
ufo = pd.concat([ufo, type_set], axis=1)


# ### Features from dates

# In[42]:


# Look at the first 5 rows of the date column
display(ufo['date'].head())

# Extract the month from the date column
ufo["month"] = ufo["date"].apply(lambda x: x.month)

# Extract the year from the date column
ufo["year"] = ufo["date"].apply(lambda x: x.year)

# Take a look at the head of all three columns
ufo[['date', 'month', 'year']].head()


# ### Text vectorization

# In[43]:


# Take a look at the head of the desc field
print(ufo['desc'].head())

# Create the tfidf vectorizer object
vec = TfidfVectorizer()

# Use vec's fit_transform method on the desc field
desc_tfidf = vec.fit_transform(ufo['desc'])

# Look at the number of columns this creates
print(desc_tfidf.shape)


# ### Selecting the ideal dataset

# In[44]:


# Check the correlation between the seconds, seconds_log, and minutes columns
print(ufo[['seconds', 'seconds_log', 'minutes']].corr())

# Make a list of features to drop
to_drop = ['city', 'country', 'lat', 'long', 'state', 'date', 'recorded', 'desc', 'seconds', 'minutes', 'length_of_time']

# Drop those features
ufo = ufo.drop(to_drop, axis = 1)
ufo.head()

# Let's also filter some words out of the text vector we created
# filtered_words = words_to_filter(vocab, vec.vocabulary_, desc_tfidf, 4)


# ### Modeling the UFO dataset, part 1

# In[45]:


ufo.isnull().sum()


# In[46]:


ufo = ufo.replace([np.inf, -np.inf], np.nan).dropna()
X = ufo[['seconds_log', 'changing', 'chevron', 'cigar', 'circle', 'cone',
       'cross', 'cylinder', 'diamond', 'disk', 'egg', 'fireball', 'flash',
       'formation', 'light', 'other', 'oval', 'rectangle', 'sphere',
       'teardrop', 'triangle', 'unknown', 'month', 'year']]
y = ufo['country_enc']


# In[47]:


# Split the X and y sets using train_test_split, setting stratify=y
train_X, test_X, train_y, test_y = train_test_split(X, y, stratify = y)

# Fit knn to the training sets
knn.fit(train_X, train_y)

# Print the score of knn on the test sets
print(knn.score(test_X, test_y))


# ### Modeling the UFO dataset, part 2

# In[2]:


# # Use the list of filtered words we created to filter the text vector
# filtered_text = desc_tfidf[:, list(filtered_words)]

# # Split the X and y sets using train_test_split, setting stratify=y 
# train_X, test_X, train_y, test_y = train_test_split(filtered_text.toarray(), y, stratify=y)

# # Fit nb to the training sets
# nb.fit(train_X, train_y)


# # ORDENANDO DATAFRAME POR UMA COLUNA
# 

# In[3]:


df.sort_values(by=['Brand'], inplace=True)
# descendente
df.sort_values(by=['Brand'], inplace=True, ascending=False)


# # ORDENANDO DATAFRAME POR MAIS DE UMA COLUNA

# In[ ]:


df.sort_values(by=['Year','Price'], inplace=True)

