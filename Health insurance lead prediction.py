#!/usr/bin/env python
# coding: utf-8

# # Health Insurance Lead Prediction -

# # Project Methodology
# FinMan Company is looking to leverage their client base by cross selling insurance products to existing customers. Insurance policies are offered to prospective and existing clients based on website landing and consumer election to fill out additional information forms. FinMan company would like to leverage their acquired information to classify positive leads for outreach programs using machine learning classifiers.When these customers fill-up the form, their Response towards the policy is considered positive and they are classified as a lead.
#         Once these leads are acquired, the sales advisors approach them to convert and thus the company can sell proposed health insurance to these leads in a more efficient manner.
# 
# # Data and Analytical Structure
# Data includes demographic features, policy features (for current customers) and example positive classifications for ML model validation and interpretation. The source can be found here. The project analysis will follow the OSEMN framework: Obtain, Scrub, Explore, Model and Interpret.

# # Data and Analytical Structure
# The project dataset is provided by Insurance company. Data includes demographic features, policy features (for current customers) and example positive classifications for ML model validation and interpretation. The source can be found by data of sample  size 50882  & columns are 14,
# 
# ##Columns of Train Data Description:
# ID - Unique Identifier for a row
# 
# City_Code - Code for the City of the customers
# 
# Region_Code - Code for the Region of the customers
# 
# Accomodation_Type - Customer Owns or Rents the house
# 
# Reco_Insurance_Type - Joint or Individual type for the recommended insurance
# 
# Upper_Age - Maximum age of the customer
# 
# Lower _Age - Minimum age of the customer
# 
# Is_Spouse - If the customers are married to each other(in case of joint insurance)
# 
# Health_Indicator - Encoded values for health of the customer
# 
# Holding_Policy_Duration Duration - (in years) of holding policy (a policy that customer has already subscribed to with the company)
# 
# Holding_Policy_Type - Type of holding policy
# 
# Reco_Policy_Cat - Encoded value for recommended health insurance
# 
# Reco_Policy_Premium - Annual Premium (INR) for the recommended health insurance
# 
# Response (Target)
# 0 : Customer did not show interest in the recommended policy
# 
# 1 : Customer showed interest in the recommended policy

# # Data & Packages | Obtain

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
pd.set_option('display.max_columns',50)


# In[10]:


from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')


# In[11]:


traindf = pd.read_csv('train.csv.zip')
testdf = pd.read_csv('test.csv.zip')
traindf


# In[12]:


testdf


# # Data Cleaning

# # Dtecting Null Values

# In[13]:


traindf.isnull().sum() # find NAN value 


# In[14]:


testdf.isnull().sum()   # find NAN value


# In[15]:


#'Policy Duration' and 'Policy Type' columns are tied to non existing. 
# Accounts and may be filled with zeros.
nonclients = ['Holding_Policy_Duration','Holding_Policy_Type']
for col in nonclients:
    traindf[col] = traindf[col].fillna(0)             


# In[16]:


testdf[col] = testdf[col].fillna(0)


# In[17]:


traindf['Holding_Policy_Duration'] = traindf['Holding_Policy_Duration'].fillna(traindf['Holding_Policy_Duration'].mode()[0])


# In[18]:


traindf['Health Indicator'] = traindf['Health Indicator'].fillna(traindf['Health Indicator'].mode()[0])


# In[19]:


testdf['Holding_Policy_Duration'] = testdf['Holding_Policy_Duration'].fillna(testdf['Holding_Policy_Duration'].mode()[0])


# In[20]:


testdf['Health Indicator'] = testdf['Health Indicator'].fillna(testdf['Health Indicator'].mode()[0])


# In[21]:


traindf.info()


# In[22]:


testdf.info()


# With many of these prospects not clearly identified as current clients, its safe to assume that null values in the 'Policy Duration' and 'Policy Type' columns are tied to non existing accounts and may be filled with zeros.

# # Feature Engineering
#     Convert to numerical: Holding_Policy_Duration
# 
#     Feature engineer long term customers
#     Convert '14+' to '15' / convert to numerical
#     Note: (after EDA) Convert to binary | Accomodation_Type, Reco_Insurance_Type, Is_Spouse

# # Categorical Features

# In[23]:


# converting 'Holding_Policy_Type' from float to int 
traindf['Long_Term_Cust'] = traindf['Holding_Policy_Duration'].apply(lambda x: 'Yes' if x == '14+' else 'No')
testdf['Long_Term_Cust'] = testdf['Holding_Policy_Duration'].apply(lambda x: 'Yes' if x == '14+' else 'No')

traindf['Holding_Policy_Duration'] = traindf['Holding_Policy_Duration'].replace('14+',15).astype(float).astype(int)
testdf['Holding_Policy_Duration'] = testdf['Holding_Policy_Duration'].replace('14+',15).astype(float).astype(int)


# In[24]:


traindf


# In[25]:


testdf


# # Renaming Features

# In[26]:


traindf.rename(columns={'Is_Spouse':'Married','Health Indicator':'Health_Indicator'},inplace=True)
testdf.rename(columns={'Is_Spouse':'Married','Health Indicator':'Health_Indicator'},inplace=True)


# In[27]:


traindf


# In[28]:


testdf


# In[29]:


traindf['Avg_Age'] = (traindf['Upper_Age'] + traindf['Lower_Age']) / 2
testdf['Avg_Age'] = (testdf['Upper_Age'] + testdf['Lower_Age']) / 2


# In[30]:


traindf


# In[31]:


testdf


# Typically, insurance products are priced and underwritten based on the age of the applicant or applicants. This is especially the case in most health insurance pricing. To reflect this and retain data, an average age feature will be created and the original two features will be dropped.

# In[32]:


# feature engineering
traindf['Prim_Prem_Ratio'] = traindf['Reco_Policy_Premium'] / traindf['Upper_Age']
testdf['Prim_Prem_Ratio'] = testdf['Reco_Policy_Premium'] / testdf['Upper_Age']


# In[33]:


traindf


# In[34]:


testdf


# # Feature Selection

# In[ ]:


#Drop features we are not going to use


# In[35]:


#dataset = dataset.drop(''ID','Region_Code','Upper_Age','Lower_Age', axis = 1)
traindf.drop(['ID','Region_Code','Upper_Age','Lower_Age'],axis=1,inplace=True)
testdf2 = testdf.copy()
testdf.drop(['ID','Region_Code','Upper_Age','Lower_Age'],axis=1,inplace=True)


# In[36]:


traindf


# In[37]:


testdf


# The unique 'ID' and 'Region Code' columns will be dropped in order to simplify the data. 'Region Code' consists of far too many categorical values which would need to be one hot encoded. The feature is dropped as the data still retains the 'City Code' feature to capture some level of geographical distinction. In addition, the upper and lower age features will be dropped being represented by average age.

# In[38]:


numcols = traindf.select_dtypes('number').columns
for col in numcols:
    traindf[col] = traindf[col].astype(int)
    


# In[39]:


testdf[col] = testdf[col].astype(int)


# In[40]:


# copy for final analysis
df = traindf.copy()


# In[41]:


df


# In[42]:


vals = {'Rented':1,'Owned':2,'Individual':1,'Joint':2,'No':0,'Yes':1}
cols = ['Accomodation_Type','Reco_Insurance_Type','Married','Long_Term_Cust']

for col in cols:
    traindf[col] = traindf[col].replace(vals)


# In[43]:


testdf[col] = testdf[col].replace(vals)


# In[44]:


traindf


# In[62]:


testdf


# Features 'Accommodation Type', 'Reco Insurance Type', 'Is Spouse' will be converted to binary 
# (0 and 1).

# In[46]:


ordinal = ['Holding_Policy_Type','Reco_Policy_Cat']
for col in ordinal:
    traindf[col] = traindf[col].astype('O')


# In[47]:


testdf[col] = testdf[col].astype('O')
testdf.info()


# In[48]:


traindf


# In[49]:


testdf


# The two feature that stand out are 'Holding Policy Type' and 'Reco Policy Cat' which are listed under numerical but most likely correspond to type and category of policy in existing customers.

# # Exploratory Data Analysis

# In[50]:


corr = traindf.corr() # analyzing correlation
fig, ax = plt.subplots(figsize=(10,10))
mask = np.triu(np.ones_like(corr, dtype=np.bool))
sns.heatmap(corr, mask=mask, square=True, annot=True, cmap='YlGnBu')
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth('1')
ax.set_title('Correlation & Heat Map', fontsize=15, fontfamily='serif')
plt.show()


# In[51]:


traindf.drop(['Married'],axis=1,inplace=True)
testdf.drop(['Married'],axis=1,inplace=True)


#      Final feature selection due to multicollinearity. 

# In[52]:


targetdf = df.groupby('Response').mean().head()
targetdf.style.background_gradient(cmap='Reds')


# Customers who elect to receive additional information typically hold existing policies longer and are classified under a larger policy category with a slightly larger premium. 

# In[53]:


fig, ax = plt.subplots(figsize=(9,6))
ax = sns.countplot(data=df[df['Holding_Policy_Type']!=0],x='Holding_Policy_Type',hue='Response',palette='mako');
for p in ax.patches:
        ax.annotate(p.get_height(),(p.get_x()+0.09, p.get_height()+75))
fig.savefig('policytypecount.jpg',dpi=200,bbox_inches='tight')


# Holding Policy Type three has the highest number of positive responses but all four of the categories have approximately 30% positive to negative client responses.

# In[54]:


fig, ax = plt.subplots(figsize=(10,6))
sns.violinplot(data=df[df['Holding_Policy_Type']!=0],x='Holding_Policy_Type',y='Avg_Age',hue='Response',palette='mako');
fig.savefig('policytypexage.jpg',dpi=200,bbox_inches='tight')


# In[55]:


traincat_vars = [var for var in traindf.columns if traindf[var].dtype == 'O']
testcat_vars = [var for var in testdf.columns if testdf[var].dtype == 'O']


# The violin plot gives an interesting take on Average Age versus Holding Policy Type. HPT 3 shows a pretty even distribution across age groups while HPT 1 is heavily made up of younger individuals.

# # Final Transformations

# In[56]:


def replace_categories(df, var, target):
    # Order variable categories | lowest to highest against target (price)
    ordered_labels = df.groupby([var])[target].mean().sort_values().index
    # Dictionary of ordered categories to integer values
    ordinal_label = {k: i for i, k in enumerate(ordered_labels, 0)}
    # Replace the categorical strings by integers using dictionary
    df[var] = df[var].map(ordinal_label)


# In[57]:


for var in traincat_vars:
    replace_categories(traindf, var, 'Avg_Age')


# In[58]:


for var in testcat_vars:
    replace_categories(testdf, var, 'Avg_Age')


# With each of the categorical values mapped to values with respect to average age, the resulting values will end up on a similar scale as the rest of the dataset. In order to minimize data manipulation for modeling, no label encoding or standard scaling will occur.

# In[59]:


# labelencoder = preprocessing.LabelEncoder()
# scaler = preprocessing.StandardScaler()


# In[60]:


# traindf['City_Code'] = labelencoder.fit_transform(traindf['City_Code'])
# traindfscaled = scaler.fit_transform(traindf)


# # Pycaret

# In[61]:


get_ipython().system(' pip install pycaret ')


# In[2]:


get_ipython().system(' pip install preprocess')


# In[63]:


import pycaret
import preprocess as preprocess
from pycaret.datasets import get_data
from pycaret.classification import *


# In[64]:


dataset = traindf.copy()
data = dataset.sample(frac=0.80, random_state=786)
data_unseen = dataset.drop(data.index).reset_index(drop=True)
print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))


# In[65]:


clf = setup(data=data,target='Response',session_id=123,numeric_features=['Long_Term_Cust','Health_Indicator','Accomodation_Type','Reco_Insurance_Type','Holding_Policy_Duration','Holding_Policy_Type'])


# In[66]:


#PREDICTION 01: By Using Logistic Regression Model
compare_models()


# # GridSearchCV

# In[67]:


get_ipython().system(' pip install xgboost')


# In[68]:


from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report


# In[69]:


def model_visuals (model, X_test, y_test):
    '''Plots the confusion matrix and ROC-AUC plot'''
    fig, axes = plt.subplots(figsize = (12, 6), ncols = 2)  # confusion matrix
    metrics.plot_confusion_matrix(model, X_test, y_test, normalize = 'true', 
                          cmap = 'Blues', ax = axes[0])
    axes[0].set_title('Confusion Matrix');
    # ROC-AUC Curve
    roc_auc = metrics.plot_roc_curve(model, X_test, y_test,ax=axes[1])
    axes[1].plot([0,1],[0,1],ls=':')
    axes[1].set_title('ROC-AUC Plot')
    axes[1].grid()
    axes[1].legend()
    fig.tight_layout()
    plt.show()


# In[70]:


traindf.info()


# In[72]:


X_train, X_test, y_train, y_test = train_test_split(traindf.drop(columns=['Response'],axis=1),traindf['Response'],test_size=0.2, random_state=42)


# In[73]:


gbclf = GradientBoostingClassifier(random_state=42)
gbclf.fit(X_train,y_train)


# In[74]:


param_grid = {
    'learning_rate': [0.1,0.2],
    'max_depth': [6],
    'subsample': [0.5,0.7,1],
    'n_estimators': [100]
}


# In[75]:


grid_clf = GridSearchCV(gbclf,param_grid,scoring='roc_auc',cv=None,n_jobs=1)
grid_clf.fit(X_train,y_train)

best_parameters = grid_clf.best_params_

print('Grid search found the following optimal parameters: ')
for param_name in sorted(best_parameters.keys()):
    print('%s: %r' % (param_name,best_parameters[param_name]))
    
training_preds = grid_clf.predict(X_train)
test_preds = grid_clf.predict(X_test)
training_accuracy = accuracy_score(y_train,training_preds)
test_accuracy = accuracy_score(y_test,test_preds)

print('')
print('Training Accuracy: {:.4}%'.format(training_accuracy*100))
print('Validation Accuracy: {:.4}%'.format(test_accuracy*100))


# Similar accuracy in the training and test sets suggests minimal under/over fitting.

# # Conclusion :
# With each of the categorical values mapped to values with respect to average age, the resulting values will end up on a similar scale as the rest of the dataset. In order to minimize data manipulation for modeling, no label encoding or standard scaling will occur.
# Holding Policy Type three has the highest number of positive responses but all four of the categories have approximately 30% positive to negative client responses.
# Similar accuracy in the training and test sets suggests minimal under/over fitting.
# 
