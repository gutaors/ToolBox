#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Basic features and readability scores

# ### Character count of Russian tweets

# In[2]:


tweets = pd.read_csv('russian_tweets.csv', index_col = 0)
tweets.head()


# In[3]:


# Create a feature char_count
tweets['char_count'] = tweets['content'].apply(lambda x: len(x))

# Print the average character count
print(tweets['char_count'].mean())


# ### Word count of TED talks

# In[4]:


ted = pd.read_csv('ted.csv')
ted.head()


# In[5]:


# Function that returns number of words in a string
def count_words(string):
	# Split the string into words
    words = string.split()
    
    # Return the number of words
    return len(words)

# Create a new feature word_count
ted['word_count'] = ted['transcript'].apply(count_words)

# Print the average word count of the talks
print(ted['word_count'].mean())


# ### Hashtags and mentions in Russian tweets

# In[6]:


# Function that returns numner of hashtags in a string
def count_hashtags(string):
	# Split the string into words
    words = string.split()
    
    # Create a list of words that are hashtags
    hashtags = [word for word in words if word.startswith('#')]
    
    # Return number of hashtags
    return(len(hashtags))

# Create a feature hashtag_count and display distribution
tweets['hashtag_count'] = tweets['content'].apply(count_hashtags)
tweets['hashtag_count'].hist()
plt.title('Hashtag count distribution')
plt.show()


# In[7]:


# Function that returns number of mentions in a string
def count_mentions(string):
	# Split the string into words
    words = string.split()
    
    # Create a list of words that are mentions
    mentions = [word for word in words if word.startswith('@')]
    
    # Return number of mentions
    return(len(mentions))

# Create a feature mention_count and display distribution
tweets['mention_count'] = tweets['content'].apply(count_mentions)
tweets['mention_count'].hist()
plt.title('Mention count distribution')
plt.show()


# ### Readability of 'The Myth of Sisyphus'

# In[8]:


sisyphus_essay = """The gods had condemned Sisyphus to ceaselessly rolling a rock to the top of a mountain, whence the stone would fall back of its own weight. They had thought with some reason that there is no more dreadful punishment than futile and hopeless labor. If one believes Homer, Sisyphus was the wisest and most prudent of mortals. According to another tradition, however, he was disposed to practice the profession of highwayman. I see no contradiction in this. Opinions differ as to the reasons why he became the futile laborer of the underworld. To begin with, he is accused of a certain levity in regard to the gods. He stole their secrets. Egina, the daughter of Esopus, was carried off by Jupiter. The father was shocked by that disappearance and complained to Sisyphus. He, who knew of the abduction, offered to tell about it on condition that Esopus would give water to the citadel of Corinth. To the celestial thunderbolts he preferred the benediction of water. He was punished for this in the underworld. Homer tells us also that Sisyphus had put Death in chains. Pluto could not endure the sight of his deserted, silent empire. He dispatched the god of war, who liberated Death from the hands of her conqueror. It is said that Sisyphus, being near to death, rashly wanted to test his wife's love. He ordered her to cast his unburied body into the middle of the public square. Sisyphus woke up in the underworld. And there, annoyed by an obedience so contrary to human love, he obtained from Pluto permission to return to earth in order to chastise his wife. But when he had seen again the face of this world, enjoyed water and sun, warm stones and the sea, he no longer wanted to go back to the infernal darkness. Recalls, signs of anger, warnings were of no avail. Many years more he lived facing the curve of the gulf, the sparkling sea, and the smiles of earth. A decree of the gods was necessary. Mercury came and seized the impudent man by the collar and, snatching him from his joys, lead him forcibly back to the underworld, where his rock was ready for him. You have already grasped that Sisyphus is the absurd hero. He is, as much through his passions as through his torture. His scorn of the gods, his hatred of death, and his passion for life won him that unspeakable penalty in which the whole being is exerted toward accomplishing nothing. This is the price that must be paid for the passions of this earth. Nothing is told us about Sisyphus in the underworld. Myths are made for the imagination to breathe life into them. As for this myth, one sees merely the whole effort of a body straining to raise the huge stone, to roll it, and push it up a slope a hundred times over; one sees the face screwed up, the cheek tight against the stone, the shoulder bracing the clay-covered mass, the foot wedging it, the fresh start with arms outstretched, the wholly human security of two earth-clotted hands. At the very end of his long effort measured by skyless space and time without depth, the purpose is achieved. Then Sisyphus watches the stone rush down in a few moments toward tlower world whence he will have to push it up again toward the summit. He goes back down to the plain. It is during that return, that pause, that Sisyphus interests me. A face that toils so close to stones is already stone itself! I see that man going back down with a heavy yet measured step toward the torment of which he will never know the end. That hour like a breathing-space which returns as surely as his suffering, that is the hour of consciousness. At each of those moments when he leaves the heights and gradually sinks toward the lairs of the gods, he is superior to his fate. He is stronger than his rock. If this myth is tragic, that is because its hero is conscious. Where would his torture be, indeed, if at every step the hope of succeeding upheld him? The workman of today works everyday in his life at the same tasks, and his fate is no less absurd. But it is tragic only at the rare moments when it becomes conscious. Sisyphus, proletarian of the gods, powerless and rebellious, knows the whole extent of his wretched condition: it is what he thinks of during his descent. The lucidity that was to constitute his torture at the same time crowns his victory. There is no fate that can not be surmounted by scorn. If the descent is thus sometimes performed in sorrow, it can also take place in joy. This word is not too much. Again I fancy Sisyphus returning toward his rock, and the sorrow was in the beginning. When the images of earth cling too tightly to memory, when the call of happiness becomes too insistent, it happens that melancholy arises in man's heart: this is the rock's victory, this is the rock itself. The boundless grief is too heavy to bear. These are our nights of Gethsemane. But crushing truths perish from being acknowledged. Thus, Edipus at the outset obeys fate without knowing it. But from the moment he knows, his tragedy begins. Yet at the same moment, blind and desperate, he realizes that the only bond linking him to the world is the cool hand of a girl. Then a tremendous remark rings out: "Despite so many ordeals, my advanced age and the nobility of my soul make me conclude that all is well." Sophocles' Edipus, like Dostoevsky's Kirilov, thus gives the recipe for the absurd victory. Ancient wisdom confirms modern heroism. One does not discover the absurd without being tempted to write a manual of happiness. "What!---by such narrow ways--?" There is but one world, however. Happiness and the absurd are two sons of the same earth. They are inseparable. It would be a mistake to say that happiness necessarily springs from the absurd. Discovery. It happens as well that the felling of the absurd springs from happiness. "I conclude that all is well," says Edipus, and that remark is sacred. It echoes in the wild and limited universe of man. It teaches that all is not, has not been, exhausted. It drives out of this world a god who had come into it with dissatisfaction and a preference for futile suffering. It makes of fate a human matter, which must be settled among men. All Sisyphus' silent joy is contained therein. His fate belongs to him. His rock is a thing. Likewise, the absurd man, when he contemplates his torment, silences all the idols. In the universe suddenly restored to its silence, the myriad wondering little voices of the earth rise up. Unconscious, secret calls, invitations from all the faces, they are the necessary reverse and price of victory. There is no sun without shadow, and it is essential to know the night. The absurd man says yes and his efforts will henceforth be unceasing. If there is a personal fate, there is no higher destiny, or at least there is, but one which he concludes is inevitable and despicable. For the rest, he knows himself to be the master of his days. At that subtle moment when man glances backward over his life, Sisyphus returning toward his rock, in that slight pivoting he contemplates that series of unrelated actions which become his fate, created by him, combined under his memory's eye and soon sealed by his death. Thus, convinced of the wholly human origin of all that is human, a blind man eager to see who knows that the night has no end, he is still on the go. The rock is still rolling. I leave Sisyphus at the foot of the mountain! One always finds one's burden again. But Sisyphus teaches the higher fidelity that negates the gods and raises rocks. He too concludes that all is well. This universe henceforth without a master seems to him neither sterile nor futile. Each atom of that stone, each mineral flake of that night filled mountain, in itself forms a world. The struggle itself toward the heights is enough to fill a man's heart. One must imagine Sisyphus happy."""


# In[9]:


# # Import Textatistic
# from textatistic import Textatistic

# # Compute the readability scores 
# readability_scores = Textatistic(sisyphus_essay).scores

# # Print the flesch reading ease score
# flesch = readability_scores['flesch_score']
# print("The Flesch Reading Ease is %.2f" % (flesch))


# ### Readability of various publications

# In[10]:


forbes = "The idea is to create more transparency about companies and individuals that are breaking the law or are non-compliant with official obligations and incentivize the right behaviors with the overall goal of improving governance and market order. The Chinese Communist Party intends the social credit score system to “allow the trustworthy to roam freely under heaven while making it hard for the discredited to take a single step.” Even though the system is still under development it currently plays out in real life in myriad ways for private citizens, businesses and government officials. Generally, higher credit scores give people a variety of advantages. Individuals are often given perks such as discounted energy bills and access or better visibility on dating websites. Often, those with higher social credit scores are able to forgo deposits on rental properties, bicycles, and umbrellas. They can even get better travel deals. In addition, Chinese hospitals are currently experimenting with social credit scores. A social credit score above 650 at one hospital allows an individual to see a doctor without lining up to pay."


# In[11]:


harvard_law = "In his important new book, The Schoolhouse Gate: Public Education, the Supreme Court, and the Battle for the American Mind, Professor Justin Driver reminds us that private controversies that arise within the confines of public schools are part of a broader historical arc — one that tracks a range of cultural and intellectual flashpoints in U.S. history. Moreover, Driver explains, these tensions are reflected in constitutional law, and indeed in the history and jurisprudence of the Supreme Court. As such, debates that arise in the context of public education are not simply about the conflict between academic freedom, public safety, and student rights. They mirror our persistent struggle to reconcile our interest in fostering a pluralistic society, rooted in the ideal of individual autonomy, with our desire to cultivate a sense of national unity and shared identity (or, put differently, our effort to reconcile our desire to forge common norms of citizenship with our fear of state indoctrination and overencroachment). In this regard, these debates reflect the unique role that both the school and the courts have played in defining and enforcing the boundaries of American citizenship. "


# In[12]:


r_digest ="This week 30 passengers were reportedly injured when a Turkish Airlines flight landing at John F. Kennedy International Airport encountered turbulent conditions. Injuries included bruises, bloody noses, and broken bones. In mid-February, a Delta Airlines flight made an emergency landing to assist three passengers in getting to the nearest hospital after some sudden and unexpected turbulence. Doctors treated 15 passengers after a flight from Miami to Buenos Aires last October for everything from severe bruising to nosebleeds after the plane caught some rough winds over Brazil. In 2016, 23 passengers were injured on a United Airlines flight after severe turbulence threw people into the cabin ceiling. The list goes on. Turbulence has been become increasingly common, with painful outcomes for those on board. And more costly to the airlines, too. Forbes estimates that the cost of turbulence has risen to over $500 million each year in damages and delays. And there are no signs the increase in turbulence will be stopping anytime soon."
time_kids = "That, of course, is easier said than done. The more you eat salty foods, the more you develop a taste for them. The key to changing your diet is to start small. “Small changes in sodium in foods are not usually noticed,” Quader says. Eventually, she adds, the effort will reset a kid’s taste buds so the salt cravings stop. Bridget Murphy is a dietitian at New York University’s Langone Medical Center. She suggests kids try adding spices to their food instead of salt. Eating fruits and veggies and cutting back on packaged foods will also help. Need a little inspiration? Murphy offers this tip: Focus on the immediate effects of a diet that is high in sodium. High blood pressure can make it difficult to be active. “Do you want to be able to think clearly and perform well in school?” she asks. “If you’re an athlete, do you want to run faster?” If you answered yes to these questions, then it’s time to shake the salt habit."


# In[13]:


# # Import Textatistic
# from textatistic import Textatistic

# # List of excerpts
# excerpts = [forbes, harvard_law, r_digest, time_kids]

# # Loop through excerpts and compute gunning fog index
# gunning_fog_scores = []
# for excerpt in excerpts:
#     readability_scores = Textatistic(excerpt).scores
#     gunning_fog = readability_scores['gunningfog_score']
#     gunning_fog_scores.append(gunning_fog)

# # Print the gunning fog indices
# print(gunning_fog_scores)


# ## Text preprocessing, POS tagging and NER
# 

# ### Tokenizing the Gettysburg Address

# In[14]:


gettysburg = "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we're engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We're met on a great battlefield of that war. We've come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It's altogether fitting and proper that we should do this. But, in a larger sense, we can't dedicate - we can not consecrate - we can not hallow - this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It's rather for us to be here dedicated to the great task remaining before us - that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion - that we here highly resolve that these dead shall not have died in vain - that this nation, under God, shall have a new birth of freedom - and that government of the people, by the people, for the people, shall not perish from the earth."


# In[15]:


import spacy

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp(gettysburg)

# Generate the tokens
tokens = [token.text for token in doc]
print(tokens)


# ### Lemmatizing the Gettysburg address

# In[16]:


import spacy

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp(gettysburg)

# Generate lemmas
lemmas = [token.lemma_ for token in doc]

# Convert lemmas into a string
print(' '.join(lemmas))


# ### Cleaning a blog post

# In[17]:


blog = "Twenty-first-century politics has witnessed an alarming rise of populism in the U.S. and Europe. The first warning signs came with the UK Brexit Referendum vote in 2016 swinging in the way of Leave. This was followed by a stupendous victory by billionaire Donald Trump to become the 45th President of the United States in November 2016. Since then, Europe has seen a steady rise in populist and far-right parties that have capitalized on Europe’s Immigration Crisis to raise nationalist and anti-Europe sentiments. Some instances include Alternative for Germany (AfD) winning 12.6% of all seats and entering the Bundestag, thus upsetting Germany’s political order for the first time since the Second World War, the success of the Five Star Movement in Italy and the surge in popularity of neo-nazism and neo-fascism in countries such as Hungary, Czech Republic, Poland and Austria."


# In[18]:


stopwords = ['fifteen', 'noone', 'whereupon', 'could', 'ten', 'all', 'please', 'indeed', 'whole', 'beside', 'therein', 'using', 'but', 'very', 'already', 'about', 'no', 'regarding', 'afterwards', 'front', 'go', 'in', 'make', 'three', 'here', 'what', 'without', 'yourselves', 'which', 'nothing', 'am', 'between', 'along', 'herein', 'sometimes', 'did', 'as', 'within', 'elsewhere', 'was', 'forty', 'becoming', 'how', 'will', 'other', 'bottom', 'these', 'amount', 'across', 'the', 'than', 'first', 'namely', 'may', 'none', 'anyway', 'again', 'eleven', 'his', 'meanwhile', 'name', 're', 'from', 'some', 'thru', 'upon', 'whither', 'he', 'such', 'down', 'my', 'often', 'whether', 'made', 'while', 'empty', 'two', 'latter', 'whatever', 'cannot', 'less', 'many', 'you', 'ours', 'done', 'thus', 'since', 'everything', 'for', 'more', 'unless', 'former', 'anyone', 'per', 'seeming', 'hereafter', 'on', 'yours', 'always', 'due', 'last', 'alone', 'one', 'something', 'twenty', 'until', 'latterly', 'seems', 'were', 'where', 'eight', 'ourselves', 'further', 'themselves', 'therefore', 'they', 'whenever', 'after', 'among', 'when', 'at', 'through', 'put', 'thereby', 'then', 'should', 'formerly', 'third', 'who', 'this', 'neither', 'others', 'twelve', 'also', 'else', 'seemed', 'has', 'ever', 'someone', 'its', 'that', 'does', 'sixty', 'why', 'do', 'whereas', 'are', 'either', 'hereupon', 'rather', 'because', 'might', 'those', 'via', 'hence', 'itself', 'show', 'perhaps', 'various', 'during', 'otherwise', 'thereafter', 'yourself', 'become', 'now', 'same', 'enough', 'been', 'take', 'their', 'seem', 'there', 'next', 'above', 'mostly', 'once', 'a', 'top', 'almost', 'six', 'every', 'nobody', 'any', 'say', 'each', 'them', 'must', 'she', 'throughout', 'whence', 'hundred', 'not', 'however', 'together', 'several', 'myself', 'i', 'anything', 'somehow', 'or', 'used', 'keep', 'much', 'thereupon', 'ca', 'just', 'behind', 'can', 'becomes', 'me', 'had', 'only', 'back', 'four', 'somewhere', 'if', 'by', 'whereafter', 'everywhere', 'beforehand', 'well', 'doing', 'everyone', 'nor', 'five', 'wherein', 'so', 'amongst', 'though', 'still', 'move', 'except', 'see', 'us', 'your', 'against', 'although', 'is', 'became', 'call', 'have', 'most', 'wherever', 'few', 'out', 'whom', 'yet', 'be', 'own', 'off', 'quite', 'with', 'and', 'side', 'whoever', 'would', 'both', 'fifty', 'before', 'full', 'get', 'sometime', 'beyond', 'part', 'least', 'besides', 'around', 'even', 'whose', 'hereby', 'up', 'being', 'we', 'an', 'him', 'below', 'moreover', 'really', 'it', 'of', 'our', 'nowhere', 'whereby', 'too', 'her', 'toward', 'anyhow', 'give', 'never', 'another', 'anywhere', 'mine', 'herself', 'over', 'himself', 'to', 'onto', 'into', 'thence', 'towards', 'hers', 'nevertheless', 'serious', 'under', 'nine']


# In[19]:


# Load model and create Doc object
nlp = spacy.load('en_core_web_sm')
doc = nlp(blog)

# Generate lemmatized tokens
lemmas = [token.lemma_ for token in doc]

# Remove stopwords and non-alphabetic tokens
a_lemmas = [lemma for lemma in lemmas 
            if lemma.isalpha() and lemma not in stopwords]

# Print string after text cleaning
print(' '.join(a_lemmas))


# ### Cleaning TED talks in a dataframe

# In[20]:


# Function to preprocess text
def preprocess(text):
  	# Create Doc object
    doc = nlp(text, disable=['ner', 'parser'])
    # Generate lemmas
    lemmas = [token.lemma_ for token in doc]
    # Remove stopwords and non-alphabetic characters
    a_lemmas = [lemma for lemma in lemmas 
            if lemma.isalpha() and lemma not in stopwords]
    
    return ' '.join(a_lemmas)
  
# Apply preprocess to ted['transcript']
ted['transcript'] = ted['transcript'].apply(preprocess)
print(ted['transcript'])


# ### POS tagging in Lord of the Flies

# In[21]:


lotf = "He found himself understanding the wearisomeness of this life, where every path was an improvisation and a considerable part of one’s waking life was spent watching one’s feet."


# In[22]:


# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp(lotf)

# Generate tokens and pos tags
pos = [(token.text, token.pos_) for token in doc]
print(pos)


# ### Counting nouns in a piece of text

# In[23]:


nlp = spacy.load('en_core_web_sm')

# Returns number of proper nouns
def proper_nouns(text, model=nlp):
  	# Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]
    
    # Return number of proper nouns
    return pos.count('PROPN')

print(proper_nouns("Abdul, Bill and Cathy went to the market to buy apples.", nlp))


# In[24]:


nlp = spacy.load('en_core_web_sm')

# Returns number of other nouns
def nouns(text, model=nlp):
  	# Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]
    
    # Return number of other nouns
    return pos.count('NOUN')

print(nouns("Abdul, Bill and Cathy went to the market to buy apples.", nlp))


# ### Noun usage in fake news

# In[25]:


headlines = pd.read_csv("fakenews.csv", index_col = 0)
headlines.head()


# In[26]:


headlines['num_propn'] = headlines['title'].apply(proper_nouns)
headlines['num_noun'] = headlines['title'].apply(nouns)

# Compute mean of proper nouns
real_propn = headlines[headlines['label'] == 'REAL']['num_propn'].mean()
fake_propn = headlines[headlines['label'] == 'FAKE']['num_propn'].mean()

# Compute mean of other nouns
real_noun = headlines[headlines['label'] == 'REAL']['num_noun'].mean()
fake_noun = headlines[headlines['label'] == 'FAKE']['num_noun'].mean()

# Print results
print("Mean no. of proper nouns in real and fake headlines are %.2f and %.2f respectively"%(real_propn, fake_propn))
print("Mean no. of other nouns in real and fake headlines are %.2f and %.2f respectively"%(real_noun, fake_noun))


# ### Named entities in a sentence

# In[27]:


# Load the required model
nlp = spacy.load('en_core_web_sm')

# Create a Doc instance 
text = 'Sundar Pichai is the CEO of Google. Its headquarters is in Mountain View.'
doc = nlp(text)

# Print all named entities and their labels
for ent in doc.ents:
    print(ent.text, ent.label_)


# ### Identifying people mentioned in a news article

# In[28]:


tc = "It’s' been a busy day for Facebook  exec op-eds. Earlier this morning, Sheryl Sandberg broke the site’s silence around the Christchurch massacre, and now Mark Zuckerberg is calling on governments and other bodies to increase regulation around the sorts of data Facebook traffics in. He’s hoping to get out in front of heavy-handed regulation and get a seat at the table shaping it."


# In[29]:


def find_persons(text):
  # Create Doc object
    doc = nlp(text)
  
  # Identify the persons
    persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
  
  # Return persons
    return persons

print(find_persons(tc))


# ## N-Gram models

# ### BoW model for movie taglines

# In[30]:


overview = pd.read_csv('movie_overviews.csv')
corpus = overview.tagline.dropna()
overview.head()


# In[31]:


# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create CountVectorizer object
vectorizer = CountVectorizer()

# Generate matrix of word vectors
bow_matrix = vectorizer.fit_transform(corpus)

# Print the shape of bow_matrix
print(bow_matrix.shape)


# ### Analyzing dimensionality and preprocessing

# In[32]:


lem_corpus = corpus.apply(preprocess)
lem_corpus.head()


# In[33]:


# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create CountVectorizer object
vectorizer = CountVectorizer()

# Generate matrix of word vectors
bow_lem_matrix = vectorizer.fit_transform(lem_corpus)

# Print the shape of bow_lem_matrix
print(bow_lem_matrix.shape)


# ### Mapping feature indices with feature names

# In[34]:


# Create CountVectorizer object
vectorizer = CountVectorizer()

# Generate matrix of word vectors
bow_matrix = vectorizer.fit_transform(corpus)

# Convert bow_matrix into a DataFrame
bow_df = pd.DataFrame(bow_matrix.toarray())

# Map the column names to vocabulary 
bow_df.columns = vectorizer.get_feature_names()

# Print bow_df
bow_df.head()


# ### BoW vectors for movie reviews

# In[35]:


from sklearn.model_selection import train_test_split
movie_reviews = pd.read_csv('movie_reviews_clean.csv')
y = movie_reviews.sentiment
X = movie_reviews.review
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
corpus = movie_reviews.review
movie_reviews.head()


# In[36]:


# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create a CountVectorizer object
vectorizer = CountVectorizer(lowercase=True, stop_words='english')

# Fit and transform X_train
X_train_bow = vectorizer.fit_transform(X_train)

# Transform X_test
X_test_bow = vectorizer.transform(X_test)

# Print shape of X_train_bow and X_test_bow
print(X_train_bow.shape)
print(X_test_bow.shape)


# ### Predicting the sentiment of a movie review

# In[37]:


from sklearn.naive_bayes import MultinomialNB
# Create a MultinomialNB object
clf = MultinomialNB()

# Fit the classifier
clf.fit(X_train_bow, y_train)

# Measure the accuracy
accuracy = clf.score(X_test_bow, y_test)
print("The accuracy of the classifier on the test set is %.3f" % accuracy)

# Predict the sentiment of a negative review
review = "The movie was terrible. The music was underwhelming and the acting mediocre."
prediction = clf.predict(vectorizer.transform([review]))[0]
print("The sentiment predicted by the classifier is %i" % (prediction))


# ### n-gram models for movie tag lines

# In[38]:


# Generate n-grams upto n=1
vectorizer_ng1 = CountVectorizer(ngram_range=(1,1))
ng1 = vectorizer_ng1.fit_transform(corpus)

# Generate n-grams upto n=2
vectorizer_ng2 = CountVectorizer(ngram_range=(1,2))
ng2 = vectorizer_ng2.fit_transform(corpus)

# Generate n-grams upto n=3
vectorizer_ng3 = CountVectorizer(ngram_range=(1, 3))
ng3 = vectorizer_ng3.fit_transform(corpus)

# Print the number of features for each model
print("ng1, ng2 and ng3 have %i, %i and %i features respectively" % (ng1.shape[1], ng2.shape[1], ng3.shape[1]))


# ### Higher order n-grams for sentiment analysis

# In[39]:


# # Define an instance of MultinomialNB 
# clf_ng = MultinomialNB()

# # Fit the classifier 
# clf_ng.fit(X_train_ng, y_train)

# # Measure the accuracy 
# accuracy = clf_ng.score(X_test_ng, y_test)
# print("The accuracy of the classifier on the test set is %.3f" % accuracy)

# # Predict the sentiment of a negative review
# review = "The movie was not good. The plot had several holes and the acting lacked panache."
# prediction = clf_ng.predict(ng_vectorizer.transform([review]))[0]
# print("The sentiment predicted by the classifier is %i" % (prediction))


# ### Comparing performance of n-gram models

# In[40]:


import time
start_time = time.time()
# Splitting the data into training and test sets
train_X, test_X, train_y, test_y = train_test_split(movie_reviews['review'], movie_reviews['sentiment'], test_size=0.5, random_state=42, stratify=movie_reviews['sentiment'])

# Generating ngrams
vectorizer = CountVectorizer(ngram_range = (1, 3))
train_X = vectorizer.fit_transform(train_X)
test_X = vectorizer.transform(test_X)

# Fit classifier
clf = MultinomialNB()
clf.fit(train_X, train_y)

# Print accuracy, time and number of dimensions
print("The program took %.3f seconds to complete. The accuracy on the test set is %.2f. The ngram representation had %i features." % (time.time() - start_time, clf.score(test_X, test_y), train_X.shape[1]))


# ## TF-IDF and similarity scores

# ### tf-idf vectors for TED talks

# In[41]:


# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Generate matrix of word vectors
tfidf_matrix = vectorizer.fit_transform(ted)

# Print the shape of tfidf_matrix
print(tfidf_matrix.shape)


# ### Computing dot product

# In[42]:


# Initialize numpy vectors
A = np.array([1,3])
B = np.array([-2, 2])

# Compute dot product
dot_prod = np.dot(A, B)

# Print dot product
print(dot_prod)


# ### Cosine similarity matrix of a corpus

# In[43]:


from sklearn.metrics.pairwise import cosine_similarity

# Initialize an instance of tf-idf Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Generate the tf-idf vectors for the corpus
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Compute and print the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(cosine_sim)


# ### Comparing linear_kernel and cosine_similarity

# In[44]:


# Record start time
start = time.time()

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Print cosine similarity matrix
print(cosine_sim)

# Print time taken
print("Time taken: %s seconds" %(time.time() - start))


# In[45]:


from sklearn.metrics.pairwise import linear_kernel
# Record start time
start = time.time()

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Print cosine similarity matrix
print(cosine_sim)

# Print time taken
print("Time taken: %s seconds" %(time.time() - start))


# ### Plot recommendation engine

# In[46]:


overview = pd.read_csv('movie_overviews.csv')
overview = overview.set_index('title')
overview.head()


# In[47]:


movie_plots = overview.overview 
indices = overview.id
overview = pd.read_csv('movie_overviews.csv')


# In[48]:


def get_recommendations(title, cosine_sim, indices):
    # Get the index of the movie that matches the title
    idx = indices[title]
    # Get the pairwsie similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores for 10 most similar movies
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return overview['title'].iloc[movie_indices]


# In[49]:


# Initialize the TfidfVectorizer 
tfidf = TfidfVectorizer(stop_words='english')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(movie_plots.values.astype('U'))

# Generate the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
 
# Generate recommendations 
print(get_recommendations('Toy Story', cosine_sim, indices))


# ### The recommender function

# In[50]:


# Generate mapping between titles and index
indices = pd.Series(ted.index, index=ted['url']).drop_duplicates()

def get_recommendations(title, cosine_sim, indices):
    # Get index of movie that matches title
    idx = indices[title]
    # Sort the movies based on the similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores for 10 most similar movies
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return ted['url'].iloc[movie_indices]


# ### TED talk recommender

# In[51]:


transcripts = ted['transcript']


# In[52]:


# Initialize the TfidfVectorizer 
tfidf = TfidfVectorizer(stop_words = 'english')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(transcripts)

# Generate the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
 
# Generate recommendations 
print(get_recommendations('https://www.ted.com/talks/al_seckel_says_our_brains_are_mis_wired\n',cosine_sim, indices))


# ### Generating word vectors

# In[53]:


sent= 'I like apples and oranges'
# Create the doc object
doc = nlp(sent)

# Compute pairwise similarity scores
for token1 in doc:
    for token2 in doc:
        print(token1.text, token2.text, token1.similarity(token2))


# ### Computing similarity of Pink Floyd songs

# In[54]:


mother = """Mother do you think they'll drop the bomb?
Mother do you think they'll like this song?
Mother do you think they'll try to break my balls?
Ooh, ah
Mother should I build the wall?
Mother should I run for President?
Mother should I trust the government?
Mother will they put me in the firing mine?
Ooh ah,
Is it just a waste of time?
Hush now baby, baby, don't you cry.
Mama's gonna make all your nightmares come true.
Mama's gonna put all her fears into you.
Mama's gonna keep you right here under her wing.
She won't let you fly, but she might let you sing.
Mama's gonna keep baby cozy and warm.
Ooh baby, ooh baby, ooh baby,
Of course mama's gonna help build the wall.
Mother do you think she's good enough, for me?
Mother do you think she's dangerous, to me?
Mother will she tear your little boy apart?
Ooh ah,
Mother will she break my heart?
Hush now baby, baby don't you cry.
Mama's gonna check out all your girlfriends for you.
Mama won't let anyone dirty get through.
Mama's gonna wait up until you get in.
Mama will always find out where you've been.
Mama's gonna keep baby healthy and clean.
Ooh baby, ooh baby, ooh baby,
You'll always be baby to me.
Mother, did it need to be so high?"""


# In[55]:


hopes = """Beyond the horizon of the place we lived when we were young
In a world of magnets and miracles
Our thoughts strayed constantly and without boundary
The ringing of the division bell had begun
Along the Long Road and on down the Causeway
Do they still meet there by the Cut
There was a ragged band that followed in our footsteps
Running before times took our dreams away
Leaving the myriad small creatures trying to tie us to the ground
To a life consumed by slow decay
The grass was greener
The light was brighter
When friends surrounded
The nights of wonder
Looking beyond the embers of bridges glowing behind us
To a glimpse of how green it was on the other side
Steps taken forwards but sleepwalking back again
Dragged by the force of some in a tide
At a higher altitude with flag unfurled
We reached the dizzy heights of that dreamed of world
Encumbered forever by desire and ambition
There's a hunger still unsatisfied
Our weary eyes still stray to the horizon
Though down this road we've been so many times
The grass was greener
The light was brighter
The taste was sweeter
The nights of wonder
With friends surrounded
The dawn mist glowing
The water flowing
The endless river
Forever and ever"""


# In[56]:


hey = """Hey you, out there in the cold
Getting lonely, getting old
Can you feel me?
Hey you, standing in the aisles
With itchy feet and fading smiles
Can you feel me?
Hey you, don't help them to bury the light
Don't give in without a fight
Hey you out there on your own
Sitting naked by the phone
Would you touch me?
Hey you with you ear against the wall
Waiting for someone to call out
Would you touch me?
Hey you, would you help me to carry the stone?
Open your heart, I'm coming home
But it was only fantasy
The wall was too high
As you can see
No matter how he tried
He could not break free
And the worms ate into his brain
Hey you, out there on the road
Always doing what you're told
Can you help me?
Hey you, out there beyond the wall
Breaking bottles in the hall
Can you help me?
Hey you, don't tell me there's no hope at all
Together we stand, divided we fall"""


# In[57]:


# Create Doc objects
mother_doc = nlp(mother)
hopes_doc = nlp(hopes)
hey_doc = nlp(hey)

# Print similarity between mother and hopes
print(mother_doc.similarity(hopes_doc))

# Print similarity between mother and hey
print(mother_doc.similarity(hey_doc))


# In[ ]:




