#!/usr/bin/env python
# coding: utf-8

# # ** Part 0: Load the datasets required for the project **

# We will load four datasets for this project. In addition to the four datasets, we will also load two lists which contain names by gender. These lists are helpful in assigning a gender to a Yelp user by their name, since gender is not available in the Yelp dataset.

# Let's first start by creating the SparkContext.

# In[1]:


import sys
sys.path.append("/opt/packages/spark/latest/python/lib/py4j-0.10.7-src.zip")
sys.path.append("/opt/packages/spark/latest/python/")
sys.path.append("/opt/packages/spark/latest/python/pyspark")
from pyspark import SparkConf, SparkContext
sc = SparkContext()
sc


# In[2]:


import json
import os
import sys
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

# helper function to load a JSON dataset from a publicly accessible url
def get_rdd_from_path(path):
        file_reader = open(path, 'r')
        str_contents = file_reader.readlines()
        json_contents = [json.loads(x.strip()) for x in str_contents]
        rdd = sc.parallelize(json_contents, numSlices=500)
        return rdd


# The first dataset we are going to load is information about Yelp businesses. The information of each business will be stored as a Python dictionary within an RDD. The dictionary consists of the following fields:
# 
# * "business_id":"encrypted business id"
# * "name":"business name"
# * "neighborhood":"hood name"
# * "address":"full address"
# * "city":"city"
# * "state":"state -- if applicable --"
# * "postal code":"postal code"
# * "latitude":latitude
# * "longitude":longitude
# * "stars":star rating, rounded to half-stars
# * "review_count":number of reviews
# * "is_open":0/1 (closed/open)
# * "attributes":["an array of strings: each array element is an attribute"]
# * "categories":["an array of strings of business categories"]
# * "hours":["an array of strings of business hours"]
# * "type": "business"

# In[3]:


# load the data about Yelp businesses in an RDD
# each RDD element is a Python dictionary parsed from JSON using json.loads()
businesses_rdd = get_rdd_from_path('/pylon5/ci5619p/benh/yelp_academic_dataset_business.json')
print (businesses_rdd.count())
print (businesses_rdd.take(2))


# The second dataset we are going to load is information about Yelp users. Each user's information will be stored as a Python dictionary within an RDD. The dictionary consists of the following fields:
# 
# *  "user_id":"encrypted user id"
# *  "name":"first name"
# *  "review_count":number of reviews
# *  "yelping_since": date formatted like "2009-12-19"
# *  "friends":["an array of encrypted ids of friends"]
# *  "useful":"number of useful votes sent by the user"
# *  "funny":"number of funny votes sent by the user"
# *  "cool":"number of cool votes sent by the user"
# *  "fans":"number of fans the user has"
# *  "elite":["an array of years the user was elite"]
# *  "average_stars":floating point average like 4.31
# *  "compliment_hot":number of hot compliments received by the user
# *  "compliment_more":number of more compliments received by the user
# *  "compliment_profile": number of profile compliments received by the user
# *  "compliment_cute": number of cute compliments received by the user
# *  "compliment_list": number of list compliments received by the user
# *  "compliment_note": number of note compliments received by the user
# *  "compliment_plain": number of plain compliments received by the user
# *  "compliment_cool": number of cool compliments received by the user
# *  "compliment_funny": number of funny compliments received by the user
# *  "compliment_writer": number of writer compliments received by the user
# *  "compliment_photos": number of photo compliments received by the user
# *  "type":"user"

# In[3]:


# load the data about Yelp users in an RDD
# each RDD element is a Python dictionary parsed from JSON using json.loads()
users_rdd = get_rdd_from_path('/pylon5/ci5619p/benh/yelp_academic_dataset_user.json')
print (users_rdd.count())
print (users_rdd.take(2))


# The third dataset we are going to load is information about business checkins reported by users on Yelp. Each checkin's information will be stored as a Python dictionary within an RDD. The dictionary consists of the following fields:
# 
# *  "checkin_info":["an array of check ins with the format day-hour:number of check ins from hour to hour+1"]
# *  "business_id":"encrypted business id"
# *  "type":"checkin"

# In[5]:


# load the data about business checkins reported by users on Yelp in an RDD
# each RDD element is a Python dictionary parsed from JSON using json.loads()
checkins_rdd = get_rdd_from_path('/pylon5/ci5619p/benh/yelp_academic_dataset_checkin.json')
print (checkins_rdd.count())
print (checkins_rdd.take(2))


# The fourth dataset we are going to load is information about business reviews written by users on Yelp. Each review's data will be stored as a Python dictionary within an RDD. The dictionary consists of the following fields:
# 
# *  "review_id":"encrypted review id"
# *  "user_id":"encrypted user id"
# *  "business_id":"encrypted business id"
# *  "stars":star rating rounded to half-stars
# *  "date":"date formatted like 2009-12-19"
# *  "text":"review text"
# *  "useful":number of useful votes received
# *  "funny":number of funny votes received
# *  "cool": number of cool review votes received
# *  "type": "review"

# In[11]:


# load the data about business reviews written by users on Yelp in an RDD, limited to businesses in Pittsburgh due to DataBricks computational limits
# each RDD element is a Python dictionary parsed from JSON using json.loads()
reviews_rdd = get_rdd_from_path('/pylon5/ci5619p/benh/yelp_academic_dataset_review_pittsburgh.json')
print (reviews_rdd.count())
print (reviews_rdd.take(2))


# Finally, we will load two lists. The first list consists of male names, and the second list consists of female names, to map Yelp user names to gender.

# In[7]:


# helper function to load a list of names from a publicly accessible url
def get_names_from_path(path):
    file_reader = open(path, 'r')
    str_contents = file_reader.readlines()
    str_contents = [x.strip() for x in str_contents]
    result = str_contents[6:]
    return result

male_names = get_names_from_path('/pylon5/ci5619p/benh/male.txt')
print('First five male names: ', male_names[:5])
print('Number of male names: ', len(male_names))

female_names = get_names_from_path('/pylon5/ci5619p/benh/female.txt')
print('First five female names: ', female_names[:5])
print('Number of female names: ', len(female_names))


# # ** Part 1: Exploratory Data Analysis **

# Performing some exploratory analysis is a great step toward understanding the data before building any statistical machine learning models on it.
# 
# Please replace `<FILL IN>` with your solution. This is the general form that exercises will take. Exercises will include an explanation of what is expected, followed by code cells where one cell will have one or more `<FILL IN>` sections.  The cell that needs to be modified will have `# TODO: Replace <FILL IN> with appropriate code` on its first line.

# In[8]:


print ('Number of businesses: ', businesses_rdd.count())
print ('Number of users: ', users_rdd.count())
print ('Number of checkins: ', checkins_rdd.count())
print ('Number of reviews: ', reviews_rdd.count())


# ** Question1: ** Print the top 5 business categories by frequency and the number of times they appear in the businesses data.

# In[10]:


# TODO: Replace <FILL IN>

from operator import add

catNames = businesses_rdd.flatMap(lambda x: [(y, 1) for y in x['categories']]) #flattens the key, assigns a value of 1 to each category

catNames.reduceByKey(add).takeOrdered(5, key = lambda x: -x[1]) # counts the frequency of keys and outputs the 5 categories in descending order of frequency


# ** Question2: ** Print the top 5 cities by frequency and the number of times they appear in the businesses data.

# In[11]:


# TODO: Replace <FILL IN>

from operator import add

cityNames = businesses_rdd.flatMap(lambda x: [(x['city'], 1)]) #flattens the key, assigns a value of 1 to each category

cityNames.reduceByKey(add).takeOrdered(5, key = lambda x: -x[1]) # counts the frequency of keys and outputs the 5 categories in descending order of frequency


# ** Question3: ** Plot the histogram of stars received by businesses.

# In[12]:


# TODO: Replace <FILL IN>

businesses_stars_counts = businesses_rdd.map(lambda x: x['stars']).collect()
plt.hist(businesses_stars_counts, bins=[x/2-0.25 for x in range(2, 12)])
plt.xlabel('Stars')
plt.ylabel('Number of Businesses')


# ** Question4: ** Plot the histogram of number of reviews received by businesses.

# In[13]:


# TODO: Replace <FILL IN>

businesses_review_counts = businesses_rdd.map(lambda x: x['review_count']).collect()

plt.hist(businesses_review_counts, bins=range(1,80))
plt.xlabel('Review Count')
plt.ylabel('Number of Businesses')


# ** Question5: ** Plot the above histogram but now on a log-log scale using `bins=range(1,1000)`. Do you see a [Power Law](https://en.wikipedia.org/wiki/Power_law) relationship in the plot? Explain your answer.
# 
# ** Answer: ** Yes, there is a power relationship in the plot because the review count decreases in the power of number of businesses, or the other way around. This is also exemplified by the linear relationship between # of businesses versus the review count in the log-log graph.

# In[14]:


# TODO: Replace <FILL IN>
plt.hist(businesses_review_counts, bins=range(1,1000))
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Log of Review Count')
plt.ylabel('Log of Number of Businesses')


# ** Question6: ** Plot the histogram of number of reviews written by users.

# In[15]:


# TODO: Replace <FILL IN>

users_review_counts = users_rdd.map(lambda x: x['review_count']).collect()

plt.hist(users_review_counts, bins=range(1,80))
plt.xlabel('Review Count')
plt.ylabel('Number of Users')


# ** Question7: ** Plot the above histogram but now on a log-log scale using `bins=range(1,1000)`. Do you see a [Power Law](https://en.wikipedia.org/wiki/Power_law) relationship in the plot? Explain your answer.
# 
# ** Answer: ** Yes, there also seems to be a power relationship between the review counts and the number of users as demonstrated by a linear relationship in the log-log graph below.

# In[16]:


# TODO: Replace <FILL IN>

plt.hist(users_review_counts, bins=range(1,1000))
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Log of Review Count')
plt.ylabel('Log of Number of Users')


# ** Question8: ** Plot the histogram of number of friends a Yelp user has.

# In[17]:


# TODO: Replace <FILL IN>

user_friend_counts = users_rdd.map(lambda x: len(x['friends'])).collect()

plt.hist(user_friend_counts, bins=range(1,80))
plt.xlabel('Number of Friends')
plt.ylabel('Number of Users')


# ** Question9: ** Plot the above histogram but now on a log-log scale. Do you see a [Power Law](https://en.wikipedia.org/wiki/Power_law) relationship in the plot? Explain your answer.
# 
# ** Answer: ** Yes, there also seems to be a power relationship between the number of users and the number of friends as demonstrated by a linear relationship in the log-log graph below.

# In[18]:


# TODO: Replace <FILL IN>

plt.hist(user_friend_counts, bins=range(1,1000))
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Log of Number of Friends')
plt.ylabel('Log of Number of Users')


# ** Question10: ** Plot the histogram of number of fans a Yelp user has.

# In[19]:


# TODO: Replace <FILL IN>

users_fan_counts = users_rdd.map(lambda x: x['fans']).collect()


plt.hist(users_fan_counts, bins=range(1,30))
plt.xlabel('Number of Fans')
plt.ylabel('Number of Users')


# ** Question11: ** Plot the above histogram but now on a log-log scale. Do you see a [Power Law](https://en.wikipedia.org/wiki/Power_law) relationship in the plot? Explain your answer.
# 
# ** Answer: ** The answer is hard to infer as to whether number of users and number of fans have a power relationship because although the log-log graph starts out as a linear relationship below (see below), it is no longer linear towards the end.

# In[20]:


# TODO: Replace <FILL IN>

plt.hist(users_fan_counts, bins=range(1,1000))
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Log of Number of Fans')
plt.ylabel('Log of Number of Users')


# ** Question12: ** Plot the histogram of number of checkins per Yelp business.

# In[21]:


# TODO: Replace <FILL IN>

business_checkin_counts = checkins_rdd.map(lambda x: sum(x['checkin_info'].values())).collect()

plt.hist(business_checkin_counts, bins=range(1,150))
plt.xlabel('Number of Checkins')
plt.ylabel('Number of Businesses')


# ** Question13: ** Plot the above histogram but now on a log-log scale using `bins=range(3,200)`. Do you see a [Power Law](https://en.wikipedia.org/wiki/Power_law) relationship in the plot? Explain your answer.
# 
# ** Answer: ** No, a power relationship does not exist between number of checkins and the number of businesses. Before it starts to decline (see graph below), the number of businesses increases with increase in number of checkins. Similarly, the uniform decline starts to scatter towards the end as well.

# In[22]:


# TODO: Replace <FILL IN>

plt.hist(business_checkin_counts, bins=range(1,1000))
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Log of Number of Checkins')
plt.ylabel('Log of Number of Businesses')


# ** Question14: ** Find the maximum value of checkins per business. Filter to obtain business IDs of businesses that had these maximum number of checkins. Fill in the code required to carry out these steps.

# In[23]:


# TODO: Replace <FILL IN>

max_checkin_count = max(checkins_rdd.map(lambda x: max(x['checkin_info'].values())).collect()) # maximum number of check in for each business
                                                                                                # of which the maximum is taken


business_ids_with_max_checkins = checkins_rdd.filter(lambda x: max(x['checkin_info'].values()) == max_checkin_count)                                     .map(lambda x: x['business_id']).collect()

len(business_ids_with_max_checkins) # only one business that has the higheset check-in of all restaurants


# In[24]:


# TODO: Replace <FILL IN>

business_names_with_max_checkins = businesses_rdd     .filter(lambda x: x['business_id'] in business_ids_with_max_checkins)     .map(lambda x: (x['name'], x['city'])).collect()
business_names_with_max_checkins


# ** Question15: ** Why do you think the above list sees much higher checkins than other businesses in the dataset?
# 
# ** Answer: ** The above list sees much higher checkins than other businesses because it's an aiport that too in Las Vegas. The airport will not only act as a layover, but it also attracts a lot of travellers from the US and around the world who come to Las Vegas. 

# ** Question16: ** Plot a histogram of the stars associated with business reviews.

# In[25]:


# TODO: Replace <FILL IN>

review_stars_counts = reviews_rdd.map(lambda x: x['stars']).collect()
plt.hist(review_stars_counts, bins=[x/2-0.25 for x in range(2, 12)])
plt.xlabel('Stars')
plt.ylabel('Number of Reviews')


# ** Question17: ** Plot a histogram of the number of reviews written per Yelp user.

# In[26]:


# TODO: Replace <FILL IN>

user_review_counts = list(reviews_rdd.map(lambda x: x['business_id']).countByValue().values())

plt.hist(user_review_counts, bins=[x for x in range(1, 15)])
plt.xlabel('Number of Reviews')
plt.ylabel('Number of Users')


# ** Question18: ** Plot a histogram of the number of reviews written per Yelp business.

# In[27]:


# TODO: Replace <FILL IN>

business_review_counts = list(reviews_rdd.map(lambda x: x['business_id']).countByValue().values())

plt.hist(business_review_counts, bins=[x for x in range(1, 100)])
plt.xlabel('Number of Reviews')
plt.ylabel('Number of Businesses')


# ** Question19: ** Plot a histogram of the number of useful votes received by Yelp reviews.

# In[28]:


# TODO: Replace <FILL IN>

review_useful_counts = reviews_rdd.map(lambda x: x['votes']['useful']).collect()

plt.hist(review_useful_counts, bins=[x for x in range(1, 10)])
plt.xlabel('Number of Useful Votes')
plt.ylabel('Number of Reviews')


# ** Question20: ** Plot a histogram of the number of funny votes received by Yelp reviews.

# In[29]:


# TODO: Replace <FILL IN>

review_funny_counts = reviews_rdd.map(lambda x: x['votes']['funny']).collect()

plt.hist(review_funny_counts, bins=[x for x in range(1, 10)])
plt.xlabel('Number of Funny Votes')
plt.ylabel('Number of Reviews')


# ** Question21: ** Plot a histogram of the number of cool votes received by Yelp reviews.

# In[30]:


# TODO: Replace <FILL IN>

review_cool_counts = reviews_rdd.map(lambda x: x['votes']['cool']).collect()

plt.hist(review_cool_counts, bins=[x for x in range(1, 10)])
plt.xlabel('Number of Cool Votes')
plt.ylabel('Number of Reviews')


# ** Question22: ** Plot a pair-plot of the number of useful, funny, and cool votes received by Yelp reviews alongwith the stars associated with the review and the length of the review.

# In[32]:


# TODO: Replace <FILL IN>

review_votes_length = reviews_rdd.map(lambda x: (x['votes']['useful'], x['votes']['funny'], x['votes']['cool'], x['stars'], len(x['text']))).collect()
review_votes_length_df = pd.DataFrame(review_votes_length, columns=['useful', 'funny', 'cool', 'stars', 'length'])
sns.pairplot(review_votes_length_df)


# ** Question23: ** Let us plot the distribution of the number of words used by males and females in their reviews. We will use the lists "male_names" and "female_names" we had created earlier for this purpose. Let's first find the user IDs associated with males and females.

# In[33]:


# TODO: Replace <FILL IN>

male_users = users_rdd.filter(lambda x: x['name'] in male_names)
female_users = users_rdd.filter(lambda x: x['name'] in female_names)

male_user_ids = male_users.map(lambda x: x['user_id']).collect()
female_user_ids = female_users.map(lambda x: x['user_id']).collect()

print (len(male_user_ids))
print (len(female_user_ids))
print (users_rdd.count())


# ** Question24: ** We can now use the user ID lists to separate the reviews into those by males and females and calculate the length of each review.

# In[34]:


# TODO: Replace <FILL IN>

male_reviews = reviews_rdd.filter(lambda x: x['user_id'] in male_user_ids).map(lambda x : x['text'])
female_reviews = reviews_rdd.filter(lambda x: x['user_id'] in female_user_ids).map(lambda x : x['text'])

male_word_count = male_reviews.map(lambda x: len(x.split(" ")))
female_word_count = female_reviews.map(lambda x: len(x.split(" ")))

print ('Male and female review length averages: ', male_word_count.mean(), female_word_count.mean())


# ** Question25: ** The code below calculates the distributions of review lengths for males and female reviewers and plots them. Do you see a marked difference between the average review length of male and female reviewers? Are there any major trends or differences between the distributions of review length of male and female reviewers?
# 
# ** Answer: ** We can infer from the distribution that the females reviewers tend to writer longer reviews than males do for any given number of reviews. Interestingly, the reverse is true as well - for any given word count, females tend to have higher number of reviews. That said, the answer is uncertain to whether the difference is statistically significant, as we have more females in the dataset.

# In[35]:


male_word_distribution = list(male_word_count.map(lambda x : (x,1)).countByKey().items())
female_word_distribution = list(female_word_count.map(lambda x : (x,1)).countByKey().items())

male_word_distribution = sorted(male_word_distribution, key=lambda x: x[0])
female_word_distribution = sorted(female_word_distribution, key=lambda x: x[0])


# In[36]:


fig, ax = plt.subplots()
ax.plot([x[0] for x in male_word_distribution], [x[1] for x in male_word_distribution], label = 'Male')
ax.plot([x[0] for x in female_word_distribution], [x[1] for x in female_word_distribution], label = 'Female')

ax.set_xlim((0, 1000))
ax.set_xticks([0, 250, 500, 750])
ax.set_xticklabels(['0', '250','500','750'])

plt.xlabel('Number of words')
plt.ylabel('Number of reviews')
plt.legend()


# # ** Part 2: Classification using tree ensemble methods **

# In this section, we will predict the number of funny votes that a review has earned, indicating how funny readers found the review.

# In[7]:


from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.linalg import Vectors, DenseVector, SparseVector
from pyspark.mllib.regression import LabeledPoint


# ** Question1: ** Fill in the necessary code to calculate word counts from text reviews below.

# In[8]:


# TODO: Replace <FILL IN>

max_words = 50000

all_reviews = reviews_rdd.map(lambda x : (x['text'], x['votes']['funny']))
word_counts = list(all_reviews.flatMap(lambda x: x[0].split(" ")).map(lambda x: (x,1)).countByKey().items())
word_counts = sorted(word_counts, key=lambda x: -x[1])

unique_words = [x[0] for x in word_counts[:max_words]]
num_unique_words = len(unique_words)
print('Number of unique words: ', num_unique_words)


# ** Question2: ** We will now construct two dictionaries - one which maps from each word to a unique integer index and the second one which maps back from the index to the word. Write the code required to do this.

# In[11]:


# TODO: Replace <FILL IN>

word_to_index_dict = {unique_words[i]:i for i in range(len(unique_words))} # corresponding word and corresponding index
index_to_word_dict = {i:unique_words[i] for i in range(len(unique_words))}


# ** Question3: ** Fill in the required code below to obtain a LabeledPoint RDD that can be used to train an mllib classifier/regressor.

# In[12]:


# TODO: Replace <FILL IN>

doc_vectors = all_reviews.map(lambda x: (x[1], x[0].split())).map(lambda x: (x[0], [word_to_index_dict[w] for w in x[1] if w in word_to_index_dict]))
doc_vectors = doc_vectors.map(lambda x: LabeledPoint(x[0], SparseVector(max_words, [(i, 1.0) for i in set(x[1])])))
print(doc_vectors.count())
print(doc_vectors.take(2))


# ** Question4: ** Randomly split the doc_vectors RDD into 80% training and 20% validation data.

# In[13]:


# TODO: Replace <FILL IN>

doc_vectors_train, doc_vectors_val = doc_vectors.randomSplit(weights = [0.8,0.2], seed = 123)


# In[14]:


del doc_vectors


# ** Question5: ** Let us implement the baseline predictor which always outputs the most common value of funny votes. Fill in appropriate code.

# In[11]:


# TODO: Replace <FILL IN>

most_common_prediction = sorted(list(doc_vectors_train.map(lambda x: (x.label,1)).countByKey().items()), key = lambda x: x[-1])[0][0]
labels_and_predictions = doc_vectors_val.map(lambda lp: (lp.label, most_common_prediction))
val_mse = labels_and_predictions.map(lambda lp: (lp[0] - lp[1]) * (lp[0] - lp[1])).sum() /    float(doc_vectors_val.count())
print('Validation Root Mean Squared Error (Baseline) = ' + str(val_mse))
print('Learned baseline prediction: ', most_common_prediction)


# In[12]:


del most_common_prediction, labels_and_predictions, val_mse


# ** Question6: ** Let us now use a Decision Tree to predict the number of funny votes. Set the maximum depth of the tree to 5 and use an appropriate impurity metric for regression.

# In[14]:


# TODO: Replace <FILL IN>

dt_model = DecisionTree.trainRegressor(doc_vectors_train, {}, impurity = "variance", maxDepth = 5)

predictions = dt_model.predict(doc_vectors_val.map(lambda x: x.features))
labels_and_predictions = doc_vectors_val.map(lambda lp: lp.label).zip(predictions)
val_mse = labels_and_predictions.map(lambda lp: (lp[0] - lp[1]) * (lp[0] - lp[1])).sum() /    float(doc_vectors_val.count())
print('Validation Root Mean Squared Error (Decision Tree) = ' + str(val_mse))
print('Learned regression tree model:')
print(dt_model.toDebugString())


# In[15]:


del dt_model


# ** Question7: ** Let us now use a Random Forest ensemble to predict the number of funny votes. Set the maximum depth of the tree to 5 and use an appropriate impurity metric for regression. Build a random forest regressor with 10 trees.

# In[16]:


# TODO: Replace <FILL IN>

rf_model = RandomForest.trainRegressor(doc_vectors_train, {}, numTrees = 10, impurity = "variance", maxDepth = 5)

predictions = rf_model.predict(doc_vectors_val.map(lambda x: x.features))
labels_and_predictions = doc_vectors_val.map(lambda lp: lp.label).zip(predictions)
val_mse = labels_and_predictions.map(lambda lp: (lp[0] - lp[1]) * (lp[0] - lp[1])).sum() /    float(doc_vectors_val.count())
print('Validation Root Mean Squared Error (Random Forest) = ' + str(val_mse))
print('Learned regression RF model:')
print(rf_model.toDebugString())


# In[17]:


del rf_model


# ** Question8: ** Let us now use a Gradient Boosting Trees (GBT) ensemble to predict the number of funny votes. Set the maximum number of iterations to 10. Does this affect the number of trees in the ensemble? Do we need to set the maximum depth of trees in the ensemble? Why or why not?
# 
# ** Answer: We were unable to run the model because of the memory issue, but we presume that the maximum number of iteration will affect the number of trees in the ensemble as GBT re-labeles any misclassified label and puts more emphasis on training instances with poor predictions on the next iteration. We likely need to set the maximum depth of trees because the model will likely overfit the data otherwise.

# In[ ]:


# TODO: Replace <FILL IN>

gb_model = GradientBoostedTrees.trainRegressor(doc_vectors_train, {}, numIterations = 10)

predictions = gb_model.predict(doc_vectors_val.map(lambda x: x.features))
labels_and_predictions = doc_vectors_val.map(lambda lp: lp.label).zip(predictions)
val_mse = labels_and_predictions.map(lambda lp: (lp[0] - lp[1]) * (lp[0] - lp[1])).sum() /    float(doc_vectors_val.count())
print('Validation Root Mean Squared Error (Gradient Boosting Trees) = ' + str(val_mse))
print('Learned regression GBT model:')
print(gb_model.toDebugString())


# ** Question9: ** Which of the four methods we tried gave the best validation RMSE results? 
# 
# ** Answer: ** As mentioned before, we were unable to run GBT, but we hypothesize that the Random Forest will have the lowest validation RMSE results as GBT will likely overfit the training data, increasing the RMSE on the validation dataset.

# # ** Part 3: Collaborative filtering for recommendation **

# In this section, we will tackle a [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering) task which can be used to recommend businesses to users based on the ratings they have already assigned to some businesses they have visited.

# In[15]:


from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating


# ** Question1: ** Let us first determine the unique user and business IDs that appear in the reviews data. This will help us build dictionaries for mapping the user/business IDs to unique integer indices. Fill in the required code to build these dictionaries below.

# In[16]:


# TODO: Replace <FILL IN>

review_user_ids = reviews_rdd.map(lambda x: x['user_id']).distinct().collect()
review_business_ids = reviews_rdd.map(lambda x: x['business_id']).distinct().collect()

user_to_index_dict = {review_user_ids[i]:i for i in range(len(review_user_ids))}
business_to_index_dict = {review_business_ids[i]:i for i in range(len(review_business_ids))}


# ** Question2: ** Next, transform each review into a rating. The Rating object takes a unique user index, a unique business index, and float-valued rating.

# In[17]:


# TODO: Replace <FILL IN>

ratings_rdd = reviews_rdd.map(lambda x: Rating(user_to_index_dict[x['user_id']], business_to_index_dict[x['business_id']], x['stars']))
print(ratings_rdd.take(2))


# ** Question3: ** Let us randomly split data into 80% train and 20% validation set.

# In[18]:


# TODO: Replace <FILL IN>

ratings_rdd_train, ratings_rdd_val = ratings_rdd.randomSplit(weights = [0.8,0.2], seed = 123)


# In[19]:


del ratings_rdd


# ** Question4: ** For a succession of ranks, we will now build an collaborative filtering algorithm using ALS (Alternating Least Squares). We will use the model to obtain train as well as validation RMSE for each rank. In the cell below, you can fill in the code to carry out the model-building, prediction, and RMSE calculation.

# In[21]:


# TODO: Replace <FILL IN>

numIterations=10
ranks = list(range(1,20)) + list(range(20, 201, 20))
train_rmses = []
val_rmses = []

for rank in ranks:
    cf_model = ALS.train(ratings_rdd_train, rank, numIterations)
    
    train_data = ratings_rdd_train.map(lambda p: (p[0], p[1]))
    predictions = cf_model.predictAll(train_data).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = ratings_rdd_train.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    train_rmse = np.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    train_rmses.append(train_rmse)
    
    val_data = ratings_rdd_val.map(lambda p: (p[0], p[1]))
    predictions = cf_model.predictAll(val_data).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = ratings_rdd_val.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    val_rmse = np.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    val_rmses.append(val_rmse)
    
    print("Root Mean Squared Error (rank={}) = Train {}, Validation {}".format(rank, train_rmse, val_rmse))


# ** Question5: ** Let us plot the train and validation RMSE versus the rank. The code below does this for you. Based on this plot, what would your choice of the rank hyperparameter be? Is this choice conclusive or do we need to conduct a more extensive hyperparameter search at larger ranks than the ones we have evaluated?
# 
# ** Answer: ** As the RMSE score on the validation dataset suggets, we can set the hypterparameter to be 4. No more externsive hyperparamater search is needed because the RMSE decrease is very marginal with higher ranks.

# In[22]:


fig, ax = plt.subplots()
ax.plot(ranks, train_rmses, label='Train RMSE')
ax.plot(ranks, val_rmses, label='Validation RMSE')

plt.xlabel('Rank')
plt.ylabel('Root Mean Squared Error')
plt.legend()


# In[23]:


del cf_model, ratings_rdd_train, ratings_rdd_val


# # ** Part 4: Topic modeling for text reviews **

# In this section, we will build and examine a Bayesian topic model named [Latent Dirichlet Allocation (LDA)](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation). The goal of textual topic modeling is to discover latent topics whose mixtures generate textual documents through a stylized probabilistic generatve model. The topics often have semantic meaning. They may be associated with various aspects discussed in the text corpus such as politics, health, education, etc. Topic models are unsupervised machine learning algorithms. Hence, the nature of discovered topics is entirely dependent of the context of your dataset.

# In[4]:


from pyspark.mllib.linalg import Vectors, DenseVector, SparseVector
from pyspark.mllib.clustering import LDA, LDAModel


# ** Question1: ** Let us create a new RDD of just textual reviews from reviews_rdd, obtain word counts, and build a list of unique words that do not include stop words. Use num_stop_words as a measure of how many of the most frequent words to filter out.

# In[5]:


# TODO: Replace <FILL IN>

# number of stopwords i.e. most frequent words to remove
# removal of stopwords such as a, the, from, etc. that occur across a vast majority of documents improves topic models
num_stop_words = 1000

all_reviews = reviews_rdd.map(lambda x : x['text'])
word_counts = list(all_reviews.flatMap(lambda x: x.lower().split()).map(lambda x: (x,1)).countByKey().items())
# sort words in descending order of frequency
word_counts = sorted(word_counts, key = lambda x: -x[1]) 

# remove stopwords
unique_words = [x[0] for x in word_counts[num_stop_words:]]
num_unique_words = len(unique_words)
print('Number of unique words: ', num_unique_words)


# In[6]:


del word_counts


# ** Question2: ** We will now construct two dictionaries - one which maps from each word to a unique integer index and the second one which maps back from the index to the word. Write the code required to do this.

# In[7]:


# TODO: Replace <FILL IN>

word_to_index_dict = {unique_words[i]:i for i in range(len(unique_words))}
index_to_word_dict = {i:unique_words[i] for i in range(len(unique_words))}


# In[8]:


del unique_words


# ** Question3: ** Construct an RDD of SparseVectors. Each SparseVector is built using the word counts of a review. Hence, the RDD of SparseVectors should be obtained as a map from the RDD of document word counts.

# In[9]:


# TODO: Replace <FILL IN>

doc_vectors = all_reviews.map(lambda x: x.lower().split()).map(lambda x: [word_to_index_dict[w] for w in x if w in word_to_index_dict])
doc_vectors = doc_vectors.map(lambda x: SparseVector(num_unique_words, [(i, 1.0) for i in set(x)]))
# zipWithIndex result needs a minor transform to be acceptable to the LDA training procedure
doc_vectors = doc_vectors.zipWithIndex().map(lambda x: [x[1], x[0]]).cache()
print(doc_vectors.count())
print(doc_vectors.take(2))


# ** Question4: ** Train an LDA model with a 100 topics and the random seed set to 42.

# In[10]:


# TODO: Replace <FILL IN>

lda_model = LDA.train(doc_vectors, k = 100, seed = 42)


# In[13]:


del doc_vectors


# ** Question5: ** Display the LDA model vocabulary size.

# In[11]:


# TODO: Replace <FILL IN>

print('Model vocabulary size: ', lda_model.vocabSize())


# ** Question6: ** Display 5 learned topics and the top 100 terms that appear in each of these topics. Assign a semantic label/meaning to each of them (e.g. food, ambience, drinks, service, etc.) You can access the topic matrix using the function topicsMatrix on the model. Do the topics learned from Yelp reviews look representative of the corpus?

# In[14]:


# TODO: Replace <FILL IN>

topics = lda_model.topicsMatrix()

for topic in range(5):
    print("Topic {}:".format(str(topic)))
    for word in range(100):
        print(topics[word][topic])


# # ** Part 5: Word2Vec for text reviews **

# In this section, we will fit a [Word2Vec](https://en.wikipedia.org/wiki/Word2vec) model to the Yelp reviews text. Word2Vec is a popular model for embedding words in Euclidean space so they can be analyzed similar to real-valued vectors. Contrary to popular belief, Word2Vec models are not deep neural models. Inspite of being shallow neural networks, they capture word associations and analogies remarkably well. 

# In[10]:


from pyspark.mllib.feature import Word2Vec
import re
pattern = re.compile('[\W_]+')


# In[12]:


review_docs = reviews_rdd.map(lambda x : x['text'].lower().split())
review_docs = review_docs.map(lambda x : [pattern.sub('', w) for w in x])
print(review_docs.take(2))


# In[13]:


del reviews_rdd


# ** Question1: ** Fit a Word2Vec model to the review_docs RDD. Set the size of embedding vectors to 10, the random seed to 42, and the number of iterations to 10.

# In[ ]:


# TODO: Replace <FILL IN>

word2vec_model = Word2Vec().setVectorSize(10).setSeed(42).setNumIterations(10).fit(review_docs)


# Let's us examine what words are closely associated with some example words. Run the cell below to see word associations. Feel free to add any additional words whose results you find interesting, but do not delete any of the words already in the list.

# In[ ]:


for word in ['salt', 'pepper', 'restaurant', 'italian', 'indian', 'chinese', 'direction', 'pittsburgh', 'burgh', 'city', 'location', 'cmu', 'pizza']:
    syms = word2vec_model.findSynonyms(word, 5)
    print('Words most similar to ', word, ' : ', [s[0] for s in syms])


# ** Question2: ** What "synonyms" in the result above give rise to perfect analogies? Are there words in the result that are spurious and not good substitutes for the originally supplied word?
# 
# ** Answer: ** We wish we were able to run the code and answer which "synonyms" in the result give rise to perfect analogies, if any.

# # ** Part 6: Frequent pattern mining using FP-Growth algorithm **

# In this section, we will mine frequent subsets of items that appear together in datapoints. This type of analysis is also known as frequent itemset mining or market basket analysis. Since the tags associated with Yelp businesses are sets, we can use them to carry out the frequent item set mining by employing the FP-Growth algorithm available in Spark.

# In[4]:


from pyspark.mllib.fpm import FPGrowth


# ** Question1: ** Fill in the required code to perform itemset mining on business categories represented as an RDD of sets. Train the FP-Growth algorithm with a minimum support parameter of 0.01 and 10 partitions.

# In[5]:


# TODO: Replace <FILL IN>

business_categories = businesses_rdd.map(lambda x: x['categories'])

fpgrowth_model = FPGrowth.train(business_categories, minSupport=0.01, numPartitions=10)
result = sorted(fpgrowth_model.freqItemsets().collect(), key=lambda x: -x[1])
for fi in result:
    if len(fi[0]) > 1:
        print(fi)


# ** Question2: ** Fill in the required code to perform itemset mining on business categories represented as an RDD of sets. Train the FP-Growth algorithm with a minimum support parameter of 0.001 and 10 partitions.

# In[6]:


# TODO: Replace <FILL IN>

fpgrowth_model = FPGrowth.train(business_categories, minSupport=0.001, numPartitions=10)
result = sorted(fpgrowth_model.freqItemsets().collect(), key=lambda x: -x[1])
for fi in result:
    if len(fi[0]) > 1:
        print(fi)


# ** Question3: ** Are all the itemsets obtained by setting minimum support 0.01 included in the itemsets obtained when we set the minimum support to 0.001?
# 
# ** Answer: ** No! With the minimum support of 0.01, the frequency was as low as 612, however, in the case of 0.001, the frequency was as low as 62.

# In[9]:


del businesses_rdd


# In[7]:


del business_categories, fpgrowth_model, result


# # ** Part 7: Bonus Analysis (if any) **

# Here, you can include any additional and insightful exploratory data analysis or machine learning tasks you have carried out in addition to the guided exploration of the dataset above. Feel free to add code/markdown cells here to present your analysis.
