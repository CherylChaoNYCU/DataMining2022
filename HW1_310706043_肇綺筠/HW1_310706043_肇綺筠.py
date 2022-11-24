#!/usr/bin/env python
# coding: utf-8

# ### Data Analysis
# 
# 1. 先做EDA了解資料大致在做什麼

# In[1]:


import seaborn as sns
sns.set_style("darkgrid")
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


# #### HW requirement 1
# 1.將資料讀取進來(可用pandas套件)

# In[2]:


battles = pd.read_csv('battles.csv')
character_death = pd.read_csv('character-deaths.csv')


# In[3]:


battles.shape


# In[4]:


battles.head()


# In[5]:


character_death.shape


# In[6]:


character_death.head()


# In[7]:


sns.countplot(x=character_death["Death Year"])
plt.title("Deaths per Year")
plt.xlabel("Death Year")
plt.ylabel("Deaths")
plt.show()


# In[8]:


plt.rcParams["figure.figsize"] = (30, 10)
sns.countplot(x=character_death["Allegiances"])
plt.title("Count Plot for Allegiances")
plt.ylabel("Count")
plt.show()


# ### Data Preprocessing
# 
# #### HW requirement 2
# 2-1. 空值以0替代

# In[9]:


character_death.isna().sum()


# In[10]:


df = character_death.fillna(0)
df.isna().sum()


# 2-2. Death Year , Book of Death , Death Chapter三者取一個，將有數值的轉成1

# In[11]:


#只留下Book of Death

cols = ['Name','Death Year','Death Chapter']
character_death = character_death.drop(cols,axis = 1)
character_death.head()


# In[12]:


#Book of Death column的數值轉換（number ->1 / NaN ->0）
#使用numpy.where (condition[, x, y])
BD = np.where(character_death['Book of Death'].isnull(),0,1)
character_death['Book of Death'] = BD
character_death['Book Intro Chapter'] = character_death['Book Intro Chapter'].fillna(0)

character_death['Book of Death']


#  2-3將Allegiances轉成dummy特徵(底下有幾種分類就會變成幾個特徵，值是0或1，本來的資料集就會再增加約20種特徵)

# In[13]:


#get_dummies 是利用pandas實現one hot encode的方式
character_death = pd.get_dummies(character_death,columns = ['Allegiances'])


# 2-4亂數拆成訓練集(75%)與測試集(25%) 
# 
# -> why亂數？防止兩者資料分布差異太大，容易overfitting

# In[14]:


from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# In[15]:


#target
dy = character_death['Book of Death']
#target "Book of Death"不能出現在訓練集裡面，需拿掉
character_death = character_death.drop('Book of Death',axis = 1)
dx = character_death #data without ground truth
train_x,test_x,train_y,test_y = train_test_split(dx, dy, random_state=100, train_size=0.75)


# #### Model Building
# 
# 3. 使用scikit-learn的DecisionTreeClassifier進行預測(可以先試著將網頁範例(iris)跑出來在使用這次作業的資料集)
# 
# 4)做出Confusion Matrix，並計算Precision, Recall, Accuracy (提示: 可使用sklearn.metrics)
# 
# * Recall(召回率) = TP/(TP+FN)
# * Precision(準確率) = TP/(TP+FP)
# * F1-score = 2 * Precision * Recall / (Precision + Recall)

# In[21]:


model = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=10, min_samples_split=5, random_state=42)
model.fit(train_x,train_y)
pred_y = model.predict(test_x)
#Calculating Accuracy
acc = model.score(test_x,test_y)
#confusion matri
confusion_matrix(test_y,pred_y)
print(classification_report(test_y, pred_y))


# In[39]:



#seq = list(range(len(character_death.index) + 1))
#output = pd.DataFrame()
#output['Character'] = seq
#output['Death'] = pred_y
#pred_y
#['Survived'] = pred_y
#submit.to_csv('submit.csv', index = False)
#output
#character_death


# 5. 產出決策樹的圖

# In[17]:


pip install graphviz


# In[20]:


import graphviz
dot_data = tree.export_graphviz(model, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("HW1_310706043_肇綺筠")
graph.view()


# ### Hyperparameters Tuning

# In[23]:


dt = tree.DecisionTreeClassifier(random_state=42)
from sklearn.model_selection import GridSearchCV

#parameters for Decision tree
params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}


# In[25]:


grid_search = GridSearchCV(estimator=dt, 
                           param_grid=params, 
                           cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")

grid_search.fit(train_x, train_y)


# In[29]:


dt_best = grid_search.best_estimator_


# In[30]:


print(classification_report(test_y, dt_best.predict(test_x)))

