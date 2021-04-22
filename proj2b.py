#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Initialize OK
from client.api.notebook import Notebook
ok = Notebook('proj2b.ok')


# # Project 2 Part B: Spam/Ham Classification
# ## Classifiers
# ### The assignment is due on Monday, April 27th at 11:59pm PST.
# 
# **Collaboration Policy**
# 
# Data science is a collaborative activity. While you may talk with others about
# the project, we ask that you **write your solutions individually**. If you do
# discuss the assignments with others please **include their names** at the top
# of your notebook.

# **Collaborators**: Priyans Desai, Anuja Lohia, Vishrut Rana, Chaitali Mandavia

# ## This Assignment
# In Project 2 Part A, you made an effort to understand the data through EDA, and did some basic feature engineering. You also built a Logistic Regression model to classify Spam/Ham emails. In Part B, you will learn how to evaluate the classifiers you built. You will also have the chance to improve your model by selecting more features.
# 
# ## Warning
# We've tried our best to filter the data for anything blatantly offensive as best as we can, but unfortunately there may still be some examples you may find in poor taste. If you encounter these examples and believe it is inappropriate for students, please let a TA know and we will try to remove it for future semesters. Thanks for your understanding!
# 
# ## Score Breakdown
# Question | Points
# --- | ---
# 6a | 1
# 6b | 1
# 6c | 2
# 6d | 2
# 6e | 1
# 6f | 3
# 7 | 6
# 8 | 6
# 9 | 15
# Total | 37

# ## Setup

# In[38]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set(style = "whitegrid", 
        color_codes = True,
        font_scale = 1.5)


# In[39]:


from utils import fetch_and_cache_gdrive
fetch_and_cache_gdrive('1SCASpLZFKCp2zek-toR3xeKX3DZnBSyp', 'train.csv')
fetch_and_cache_gdrive('1ZDFo9OTF96B5GP2Nzn8P8-AL7CTQXmC0', 'test.csv')

original_training_data = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Convert the emails to lower case as a first step to processing the text
original_training_data['email'] = original_training_data['email'].str.lower()
test['email'] = test['email'].str.lower()

original_training_data.head()

from sklearn.model_selection import train_test_split

train, val = train_test_split(original_training_data, test_size=0.1, random_state=42)


# The following code is adapted from Part A of this project. You will be using it again in Part B.

# In[40]:


def words_in_texts(words, texts):
    '''
    Args:
        words (list-like): words to find
        texts (Series): strings to search in
    
    Returns:
        NumPy array of 0s and 1s with shape (n, p) where n is the
        number of texts and p is the number of words.
    '''
    indicator_array = 1 * np.array([texts.str.contains(word) for word in words]).T
    return indicator_array

some_words = ['drug', 'bank', 'prescription', 'memo', 'private']

X_train = words_in_texts(some_words, train['email']) 
Y_train = np.array(train['spam'])

X_train[:5], Y_train[:5]


# Recall that you trained the following model in Part A.

# In[41]:


from sklearn.linear_model import LogisticRegression

model =  LogisticRegression()
model.fit(X_train, Y_train)

training_accuracy = model.score(X_train, Y_train)
print("Training Accuracy: ", training_accuracy)


# ## Evaluating Classifiers

# The model you trained doesn't seem too shabby! But the classifier you made above isn't as good as this might lead us to believe. First, we are evaluating accuracy on the training set, which may provide a misleading accuracy measure, especially if we used the training set to identify discriminative features. In future parts of this analysis, it will be safer to hold out some of our data for model validation and comparison.
# 
# Presumably, our classifier will be used for **filtering**, i.e. preventing messages labeled `spam` from reaching someone's inbox. There are two kinds of errors we can make:
# - False positive (FP): a ham email gets flagged as spam and filtered out of the inbox.
# - False negative (FN): a spam email gets mislabeled as ham and ends up in the inbox.
# 
# These definitions depend both on the true labels and the predicted labels. False positives and false negatives may be of differing importance, leading us to consider more ways of evaluating a classifier, in addition to overall accuracy:
# 
# **Precision** measures the proportion $\frac{\text{TP}}{\text{TP} + \text{FP}}$ of emails flagged as spam that are actually spam.
# 
# **Recall** measures the proportion $\frac{\text{TP}}{\text{TP} + \text{FN}}$ of spam emails that were correctly flagged as spam. 
# 
# **False-alarm rate** measures the proportion $\frac{\text{FP}}{\text{FP} + \text{TN}}$ of ham emails that were incorrectly flagged as spam. 
# 
# The following image might help:
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Precisionrecall.svg/700px-Precisionrecall.svg.png" width="500px">
# 
# Note that a true positive (TP) is a spam email that is classified as spam, and a true negative (TN) is a ham email that is classified as ham.

# ### Question 6a
# 
# Suppose we have a classifier `zero_predictor` that always predicts 0 (never predicts positive). How many false positives and false negatives would this classifier have if it were evaluated on the training set and its results were compared to `Y_train`? Fill in the variables below (answers can be hard-coded):
# 
# *Tests in Question 6 only check that you have assigned appropriate types of values to each response variable, but do not check that your answers are correct.*
# 
# <!--
# BEGIN QUESTION
# name: q6a
# points: 1
# -->

# In[42]:


zero_predictor_fp = 0
zero_predictor_fn = sum(train['spam'])


# In[43]:


ok.grade("q6a");


# ### Question 6b
# 
# What are the accuracy and recall of `zero_predictor` (classifies every email as ham) on the training set? Do **NOT** use any `sklearn` functions.
# 
# <!--
# BEGIN QUESTION
# name: q6b
# points: 1
# -->

# In[44]:


zero_predictor_acc = sum(Y_train == 0)/len(Y_train)
zero_predictor_recall = 0/(0 + zero_predictor_fn)


# In[45]:


ok.grade("q6b");


# ### Question 6c
# 
# Provide brief explanations of the results from 6a and 6b. Explain why the number of false positives, number of false negatives, accuracy, and recall all turned out the way they did.
# 
# <!--
# BEGIN QUESTION
# name: q6c
# manual: True
# points: 2
# -->
# <!-- EXPORT TO PDF -->

# Since the zero_precitor classified every single email as ham, every spam email was classeified as ham. Thus, the false negatives was equal to the number of spam emails. Thus, the accuracy was 74.4.% and the recall rate 0% since the number of false negatives was 0 and every email was classified as a ham email. 

# ### Question 6d
# 
# Compute the precision, recall, and false-alarm rate of the `LogisticRegression` classifier created and trained in Part A. Do **NOT** use any `sklearn` functions.
# 
# **Note: In lecture we used the `sklearn` package to compute the rates. Here you should work through them using just the definitions to help build a deeper understanding.**
# 
# <!--
# BEGIN QUESTION
# name: q6d
# points: 2
# -->

# In[46]:


Y_train_hat = model.predict(X_train)
new_ds = train[['spam']]
new_ds['Y_train_hat'] = Y_train_hat
tp0 = new_ds[new_ds['spam'] == 1]
tp = tp0[tp0['Y_train_hat'] == 1]
fp0 = new_ds[new_ds['spam'] == 0]
fp = fp0[fp0['Y_train_hat'] == 1]
fn0 = new_ds[new_ds['spam'] == 1]
fn = fn0[fn0['Y_train_hat'] == 0]
tn0 = new_ds[new_ds['spam'] == 0]
tn = tn0[tn0['Y_train_hat'] == 0]
logistic_predictor_precision = len(tp['spam'])/(len(tp['spam']) + len(fp['spam']))
logistic_predictor_recall = len(tp['spam'])/(len(tp['spam']) + len(fn['spam']))
logistic_predictor_far = len(fp['spam'])/(len(fp['spam']) + len(tn['spam']))


# In[47]:


ok.grade("q6d");


# ### Question 6e
# 
# Are there more false positives or false negatives when using the logistic regression classifier from Part A?
# 
# <!--
# BEGIN QUESTION
# name: q6e
# manual: True
# points: 1
# -->
# <!-- EXPORT TO PDF -->

# When using the logistic regression classifier from Part A, there are more false negatives than there are false positives.

# ### Question 6f
# 
# 1. Our logistic regression classifier got 75.8% prediction accuracy (number of correct predictions / total). How does this compare with predicting 0 for every email?
# 1. Given the word features we gave you above, name one reason this classifier is performing poorly. Hint: Think about how prevalent these words are in the email set.
# 1. Which of these two classifiers would you prefer for a spam filter and why? Describe your reasoning and relate it to at least one of the evaluation metrics you have computed so far.
# 
# <!--
# BEGIN QUESTION
# name: q6f
# manual: True
# points: 3
# -->
# <!-- EXPORT TO PDF -->

# 1. The 0 predictor had an accuracy of 74.4% which is only 1% lower than the prediction from our logistic regression classifier. 
# 2. One of the reasons that the classifier is performing poorly is that the word features above seem to be prevalent in both the spam and ham emails and it is also possible that some words might not be in any of the emails. Thus, it might not classify it correctly.  
# 3. I would rather have the zero predictor than the linear regression for the spam filter since it has a lower false positive rate. This would ensure than no ham emails, which might contain important information for someone, would get filtered out. Further, the 0 predictor is more reliable since we know how it would classify all values. There is lesser uncertainty.

# ## Moving Forward
# 
# With this in mind, it is now your task to make the spam filter more accurate. In order to get full credit on the accuracy part of this assignment, you must get at least **88%** accuracy on the test set. To see your accuracy on the test set, you will use your classifier to predict every email in the `test` DataFrame and upload your predictions to Kaggle.
# 
# **Kaggle limits you to four submissions per day**. This means you should start early so you have time if needed to refine your model. You will be able to see your accuracy on the entire set when submitting to Kaggle (the accuracy that will determine your score for question 9).
# 
# Here are some ideas for improving your model:
# 
# 1. Finding better features based on the email text. Some example features are:
#     1. Number of characters in the subject / body
#     1. Number of words in the subject / body
#     1. Use of punctuation (e.g., how many '!' were there?)
#     1. Number / percentage of capital letters 
#     1. Whether the email is a reply to an earlier email or a forwarded email
# 1. Finding better (and/or more) words to use as features. Which words are the best at distinguishing emails? This requires digging into the email text itself. 
# 1. Better data processing. For example, many emails contain HTML as well as text. You can consider extracting out the text from the HTML to help you find better words. Or, you can match HTML tags themselves, or even some combination of the two.
# 1. Model selection. You can adjust parameters of your model (e.g. the regularization parameter) to achieve higher accuracy. Recall that you should use cross-validation to do feature and model selection properly! Otherwise, you will likely overfit to your training data.
# 
# You may use whatever method you prefer in order to create features, but **you are not allowed to import any external feature extraction libraries**. In addition, **you are only allowed to train logistic regression models**. No random forests, k-nearest-neighbors, neural nets, etc.
# 
# We have not provided any code to do this, so feel free to create as many cells as you need in order to tackle this task. However, answering questions 7, 8, and 9 should help guide you.
# 
# ---
# 
# **Note:** *You should use the **validation data** to evaluate your model and get a better sense of how it will perform on the Kaggle evaluation.*
# 
# ---

# In[48]:


#Creating a DataSet with all of the wanted features in this cell.


#A function which takes a character and an array of texts and determines whether there is a 
def char_finder(char, texts):
    indicator_array_intermediate = []
    texts = texts.fillna("")
    for text in texts:
        if char in text:
            indicator_array_intermediate += [1]
        else:
            indicator_array_intermediate += [0]
    indicator_array = np.array(indicator_array_intermediate)
    return indicator_array
    
    
def creating_processed_df(data):
    data['subject'] = data['subject'].fillna("")
    data['email'] = data['email'].fillna("")
    
    #len_body_characters= []
    #for i in data.index:
    #    len_body_characters.append(len(data['email'][i]))
    
    #len_subject_characters = []
    #for i in data.index:
    #    len_subject_characters.append(len(data['subject'][i]))
    
    #len_body_words = []
    #for i in data.index:
    #    words_in_body = data['email'][i].split()
    #    len_body_words.append(len(words_in_body))
        
    #len_subject_words = []
   # for i in data.index:
   #     words_in_subject = data['subject'][i].split()
    #    len_subject_words.append(len(words_in_subject))

    ex_subj = char_finder("!", data['subject'])
    opencar_body = char_finder("<", data['email'])
    buy_subj = char_finder("buy", data['subject'])
    dollar_body = char_finder("$", data['email'])
    bankruptcy = char_finder("bankruptcy", data['email'])
    body_body = char_finder("<body>", data['email'])
    html_body = char_finder("<html>", data['email'])
    now_subj = char_finder("now!", data['subject'])
    affordable_body = char_finder("affordable", data['email'])
    affordable_subj = char_finder("affordable", data['subject'])
    amazing_body = char_finder("amazing", data['email'])
    hashh = char_finder("##", data['email'])
    closecar = char_finder(">", data['email'])
    business = char_finder("business", data['subject'])
    thousands = char_finder(",000", data['email'])
    free = char_finder("free", data['email'])
    guarantee = char_finder("guarantee", data['email'])
    superr = char_finder("super", data['email'])
    million = char_finder("million", data['email'])
    billion = char_finder("billion", data['email'])
    news = char_finder("news", data['email'])
    click = char_finder("click", data['email'])
    download = char_finder("download", data['email'])
    sign_up = char_finder("sign up", data['email'])
    income = char_finder("income", data['email'])
    
    dataa = {"subject": data['subject'],
             "email": data['email'],
            "!":ex_subj,
            "<": opencar_body,
            "buy": buy_subj,
            "$": dollar_body,
            "bankruptcy": bankruptcy,
            "<body>": body_body,
            "<html>": html_body,
            "now": now_subj,
            "affordable": affordable_body,
            "amazing_body": amazing_body,
            "#": hashh,
            ">": closecar,
            "business": business,
            "thousands": thousands, 
            "free": free, 
            'guarantee': guarantee,
            "super": superr,
            "million": million, 
            "billion": billion,
            "news": news,
            "click": click,
            "download": download}
    
    features_df = pd.DataFrame(dataa)
    
    return features_df

def my_model_acc(data):
    Y_train = data['spam']
    processed = creating_processed_df(data)
    X_train = processed[["#", "!", "<", "$", "bankruptcy", "<body>", "<html>", "now", "affordable", "click", "download",
                        "amazing_body", ">", "business", "thousands", "free", "guarantee", "super", "million", "billion", "news"]]
    my_model = LogisticRegression()
    my_model.fit(X_train, Y_train)
    return my_model


# ### Question 7: Feature/Model Selection Process
# 
# In the following cell, describe the process of improving your model. You should use at least 2-3 sentences each to address the follow questions:
# 
# 1. How did you find better features for your model?
# 2. What did you try that worked / didn't work?
# 3. What was surprising in your search for good features?
# 
# <!--
# BEGIN QUESTION
# name: q7
# manual: True
# points: 6
# -->
# <!-- EXPORT TO PDF -->

# 1. For my model, I first tried using features that were suggested by the course staff. I then tried creating graphs depending on the feature I was using to determine whether this feature was actually relevant to my analysis. For example, I tried plotting graphs depicting the frequency of the words which were in ham and spam, as we did in part 2(a) of the project. I also did the same for punctuation. Then, I also tried using wordclouds. Furthermore, I also started this process by trying to google more words which occured in spam emails (https://www.codemefy.com/2018/06/11/spam-words-list/).  
# 2. I tried calculating the length of the subject and email. I thought that this would help discern between spam and ham. However, it did not work. Further, surprisingly, strongly positive words or punctuations such as "!" worked the best in classifying the email as spam or ham more accurately. 
# 3. When looking for good features, I was surprised to find that some words, that seem discerning, for example "buy", were barely present in my spam or ham emails and thus reduced the accuracy of my classifier. Furthermore, in part 2(a) of this porject, we also observed that the lengths of spam and ham emails differed quite a bit. However, this feature, suprisingly, did not help me find more relevant features.

# ### Question 8: EDA
# 
# In the cell below, show a visualization that you used to select features for your model. Include
# 
# 1. A plot showing something meaningful about the data that helped you during feature selection, model selection, or both.
# 2. Two or three sentences describing what you plotted and its implications with respect to your features.
# 
# Feel to create as many plots as you want in your process of feature selection, but select one for the response cell below.
# 
# **You should not just produce an identical visualization to question 3.** Specifically, don't show us a bar chart of proportions, or a one-dimensional class-conditional density plot. Any other plot is acceptable, as long as it comes with thoughtful commentary. Here are some ideas:
# 
# 1. Consider the correlation between multiple features (look up correlation plots and `sns.heatmap`). 
# 1. Try to show redundancy in a group of features (e.g. `body` and `html` might co-occur relatively frequently, or you might be able to design a feature that captures all html tags and compare it to these). 
# 1. Visualize which words have high or low values for some useful statistic.
# 1. Visually depict whether spam emails tend to be wordier (in some sense) than ham emails.

# Generate your visualization in the cell below and provide your description in a comment.
# 
# <!--
# BEGIN QUESTION
# name: q8
# manual: True
# format: image
# points: 6
# -->
# <!-- EXPORT TO PDF format:image -->

# In[49]:


# Write your description (2-3 sentences) as a comment here:
# At first, I tried using the length of the subject and emails in order to find out whether there was correlation between those and th esubject being spam or ham.
# However, I soon realised that certain words occured more frequently in spam emails than in ham emails and vice versa. The same was also applicable to subject lines.
# Thus, I then plotted wordclouds in order to find more helpful words/punctuation. I have included one wordcloud below, which shows the wordcloud for certain chosen features in spam emails. 

# Write the code to generate your visualization here:
from wordcloud import WordCloud

processed_data_for_wc = creating_processed_df(train)
processed_data_for_wc['spam'] = train['spam']
processed_data_for_wc = processed_data_for_wc[processed_data_for_wc['spam'] == 1]
processed_data_for_wc = processed_data_for_wc[["#", "<", "$", "bankruptcy", "<body>", "<html>", "affordable", "click", "download",
                        "amazing_body", ">", "thousands", "free", "guarantee", "super", "million", "billion", "news"]]
seriess = processed_data_for_wc.sum(axis = 0)

wordcloud = WordCloud(width = 400, height = 500, background_color = "black").fit_words(seriess)

plt.imshow(wordcloud)
# Note: if your plot doesn't appear in the PDF, you should try uncommenting the following line:
#plt.show()


# # Question 9: Submitting to Kaggle
# 
# The following code will write your predictions on the test dataset to a CSV, which you can submit to Kaggle. You may need to modify it to suit your needs.
# 
# Save your predictions in a 1-dimensional array called `test_predictions`. *Even if you are not submitting to Kaggle, please make sure you've saved your predictions to `test_predictions` as this is how your score for this question will be determined.*
# 
# Remember that if you've performed transformations or featurization on the training data, you must also perform the same transformations on the test data in order to make predictions. For example, if you've created features for the words "drug" and "money" on the training data, you must also extract the same features in order to use scikit-learn's `.predict(...)` method.
# 
# You should submit your CSV files to https://www.kaggle.com/t/c76d80f7d3204159865a324ec2936f18
# 
# **Note: You may submit up to 4 times a day. If you have submitted 4 times on a day, you will need to wait until the next day for more submissions.**
# 
# Note that this question is graded on an absolute scale based on the accuracy your model achieves on the test set and the score does not depend on your ranking on Kaggle. 
# 
# *The provided tests check that your predictions are in the correct format, but you must submit to Kaggle to evaluate your classifier accuracy.*
# 
# <!--
# BEGIN QUESTION
# name: q9
# points: 15
# -->

# In[55]:


processed_test_data = creating_processed_df(test)
X_test = processed_test_data[["#", "!", "<", "$", "bankruptcy", "<body>", "<html>", "now", "affordable", "click", "download",
                        "amazing_body", ">", "business", "thousands", "free", "guarantee", "super", "million", "billion", "news"]]
test_predictions = my_model_acc(train).predict(X_test)


# In[51]:


ok.grade("q9");


# The following saves a file to submit to Kaggle.

# In[54]:


from datetime import datetime

# Assuming that your predictions on the test set are stored in a 1-dimensional array called
# test_predictions. Feel free to modify this cell as long you create a CSV in the right format.

# Construct and save the submission:
submission_df = pd.DataFrame({
    "Id": test['id'], 
    "Class": test_predictions,
}, columns=['Id', 'Class'])
timestamp = datetime.isoformat(datetime.now()).split(".")[0]
submission_df.to_csv("submission_{}.csv".format(timestamp), index=False)

print('Created a CSV file: {}.'.format("submission_{}.csv".format(timestamp)))
print('You may now upload this CSV file to Kaggle for scoring.')


# # Submit
# Make sure you have run all cells in your notebook in order before running the cell below, so that all images/graphs appear in the output.
# **Please save before submitting!**
# 
# <!-- EXPECT 5 EXPORTED QUESTIONS -->

# In[56]:


# Save your notebook first, then run this cell to submit.
import jassign.to_pdf
jassign.to_pdf.generate_pdf('proj2b.ipynb', 'proj2b.pdf')
ok.submit()


# In[ ]:




