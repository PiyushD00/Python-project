#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_wine

alcohol = load_wine()

alcohol.keys()
x = alcohol['data']
y = alcohol['target']

print(x.shape)
print(y.shape)


# In[2]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[3]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)


# In[4]:


y_pred = model.predict(x_test)
y_pred


# In[6]:


import pandas as pd

df = pd.DataFrame(x,columns=alcohol['feature_names'])

df['target'] = y

df.sample()


# In[7]:


alcohol['target_names']


# In[8]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
cm


# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(cm,annot=True)
plt.show()


# In[10]:


accuracy  = accuracy_score(y_test, y_pred)
accuracy


# In[11]:


cr = classification_report(y_test, y_pred)
print(cr)


# In[ ]:




