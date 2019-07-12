#!/usr/bin/env python
# coding: utf-8

# In[1]:


## usefull libraries
import itertools
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tabulate
from IPython.display import HTML, display
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


def preprocessing(train_data,test_data,store): 
    """Processes data for model
    INPUT: Three dataFrames
    OUTPUT: Merged and processed train and test sets
    """
    ## dropping duplicates: 
    train_data = train_data.drop_duplicates()
    store = store.drop_duplicates()
    
    ## There is missing values in Open variable of the test set 
    ## We assume store open, if not provided
    test_data.fillna(1, inplace=True)
    
    ## dealing with missing values in store set : 
    ## All this fillna approches where explained and justified in the EDA notebook
    store['CompetitionDistance'].fillna(store['CompetitionDistance'].quantile(), inplace=True)
    store["CompetitionOpenSinceMonth"].fillna(0, inplace=True)
    store["CompetitionOpenSinceYear"].fillna(0, inplace=True)
    store["Promo2SinceWeek"].fillna(0, inplace=True)
    store["Promo2SinceYear"].fillna(0, inplace=True)
    store["PromoInterval"].fillna(0, inplace=True)
    
    ## We analyse only open stores since a close store yield a profit of 0.
    ## There're 172817 closed stores in the data. It is about 10% of the total amount of observations.
    ## To avoid any biased forecasts we will drop these values.
    train_data = train_data[train_data["Open"] != 0]
    
    ## Use only Sales bigger then zero. there are opened store with no sales on working days.
    ## There're only 54 days in the data, so we can assume that there were external factors involved, 
    ## for example manifestations.
    train_data = train_data[train_data["Sales"] > 0]
    
    ## find and drop outliers : 
    def find_low_high(df,feature):
        # find store specific Q - 3*IQR and T + 3*IQR
        #interquatile range:
        IQR = df.groupby('Store')[feature].quantile(0.75)-df.groupby('Store')[feature].quantile(0.25)
        #the quartile:
        Q = df.groupby('Store')[feature].quantile(0.25)
        # the tertiles:
        T = df.groupby('Store')[feature].quantile(0.75)

        low = Q - 3*IQR
        high = T + 3*IQR

        low_df=pd.DataFrame(columns=['low'])
        low_df['low']= low
        high_df=pd.DataFrame(columns=['high'])
        high_df['high']= high
        low_df = low_df.reset_index()
        high_df = high_df.reset_index()

        return low_df,high_df

    def find_outlier_index(df,feature):
        main_data = df[['Store',feature]]
        low = find_low_high(df,feature)[0]
        high = find_low_high(df,feature)[1]

        new_low = pd.merge(main_data, low, on='Store', how='left')
        new_low['outlier_low'] = (new_low[feature] < new_low['low'])
        index_low = new_low[new_low['outlier_low'] == True].index
        index_low = list(index_low)

        new_high = pd.merge(main_data, high, on='Store', how='left')
        new_high['outlier_high'] = new_high[feature] > new_high['high']
        index_high = new_high[new_high['outlier_high'] == True].index
        index_high = list(index_high)

        index_low.extend(index_high)
        index = list(set(index_low))
        return index

    ## finding and dropping outliers in the train_data set :
    train_data=train_data.reset_index()
    train_data.drop(find_outlier_index(train_data,"Sales"), inplace=True, axis=0)
    train_data.drop(columns='index',inplace=True)
    
    ## Join train and test with store 
    train = pd.merge(train_data, store, on='Store')
    test = pd.merge(test_data, store, on='Store')
    
    
    return train,test


# In[10]:


def feature_engineering(train,set_type='train'): 
    """ feature engineering for model
    INPUT:  DataFrame
    OUTPUT: Dataframe with more gathered features
    """
    
    ## Explore date column  
    train['Year'] = train.Date.dt.year
    train['Month'] = train.Date.dt.month
    train['Day'] = train.Date.dt.day
    train['DayOfWeek'] = train.Date.dt.dayofweek
    train['WeekOfYear'] = train.Date.dt.weekofyear
    

    ## For the StateHoliday: it is not very important to distinguish between types (0,a,b,c),
    ## they can be merged in a binary variable
    train["StateHoliday"] = train['StateHoliday'].map({"0": 0, "a": 1, "b": 1, "c": 1})

    ## Map PromoInterval variable
    train["PromoInterval"] = train['PromoInterval'].map({"0": 0, 'Jan,Apr,Jul,Oct': 1, 'Feb,May,Aug,Nov': 2,
                                                         'Mar,Jun,Sept,Dec': 3})
    ## Get dummies for categoricals
    train = pd.get_dummies(train, columns = ['StoreType','Assortment'])

    ## We can add variables describing the period of time during which competition and promotion were opened
    # Calculate competition open time in months
    train['CompetitionOpen'] = 12 * (train.Year - train.CompetitionOpenSinceYear) + (train.Month - train.CompetitionOpenSinceMonth)
    train['CompetitionOpen'] = train.CompetitionOpen.apply(lambda x: x if x > 0 else 0)
    train.drop(['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'], axis = 1, 
             inplace = True)

    # calculate Promo2 open time in months
    train['PromoOpen'] = 12 * (train.Year - train.Promo2SinceYear) + (train.WeekOfYear - train.Promo2SinceWeek) / float(4)
    train['PromoOpen'] = train.PromoOpen.apply(lambda x: x if x > 0 else 0)

    # Promo 2 is on going now
    train['Promo2_ongoing'] = ((train.Promo2 == 1) & (train.Year > train.Promo2SinceYear)).astype(int) +     ((train.Promo2 == 1) & (train.Year == train.Promo2SinceYear) & (train.WeekOfYear >= train.Promo2SinceWeek)).astype(int)
    train['Promo2_ongoing_now'] = ((train['Promo2_ongoing'] == 1) & (train.Month % 3 == train.PromoInterval % 3)).astype(int)

    # drop unuseful variables
    train.drop(['Promo2','Promo2SinceWeek','Promo2SinceYear','PromoInterval','Promo2_ongoing'], axis = 1, 
             inplace = True)

    ## *********** For analysis purposes, we may need the below features  for the train set ***********
    
    if set_type =='train': 
        ##  Sales and Customers  are likely correlated vaiables in the dataset,
        ## they can be combined into a new feature
        train['SalePerCustomer'] = train['Sales']/train['Customers']

        ## Sales ratio : 
        # this either has to be median over median (if you want to remove outliers) or it should be mean of the ratio. 
        # Median of the ratio is also an alternative
        aggregation = train[["Store","Sales","Customers"]].groupby("Store").sum()
        sales_customer_ratio = (aggregation.Sales/aggregation.Customers).to_frame(name="ratio_sales_customer").reset_index()
        train = train.merge(sales_customer_ratio,how="left",on="Store")

        # We will extract the statistics (mean and std) related to each store and add them to our dataset 
        # under the names of sales_mean and sales_std
        mean = train.groupby('Store')['Sales'].mean()
        std = train.groupby('Store')['Sales'].std()
        df_mean = pd.DataFrame(mean).reset_index()
        df_mean.rename(columns={"Sales": "sales_mean"} , inplace=True)
        df_std = pd.DataFrame(std).reset_index()
        df_std.rename(columns={"Sales": "sales_std"} , inplace=True)
        train = pd.merge(train , df_mean , on='Store' , how='left')
        train = pd.merge(train , df_std , on='Store' , how='left')


        # Once we have generated the above features, we will group the stores according to their sales level
        Q1 = train.sales_mean.quantile(0.25)
        Q2 = train.sales_mean.quantile(0.5)
        Q3 = train.sales_mean.quantile(0.75)
        train['StoreGroup1'] = (train.sales_mean < Q1).astype(int)
        train['StoreGroup2'] = ((train.sales_mean >= Q1) & (train.sales_mean < Q2)).astype(int)
        train['StoreGroup3'] = ((train.sales_mean >= Q2) & (train.sales_mean < Q3)).astype(int)
        train['StoreGroup4'] = (train.sales_mean >= Q3).astype(int)

        # Since the values of 'StoreGroup' features are either 0 or 1, we will regroup them into one feature
        train['StoreGroup']= train['StoreGroup1'] + 2*train['StoreGroup2'] + 3*train['StoreGroup3'] + 4*train['StoreGroup4']

        # We will drop the unusefull comumns
        train.drop(['StoreGroup1','StoreGroup2','StoreGroup3','StoreGroup4'],axis=1, inplace=True)


        # A log transformation could also be usefull
        train['log_sales'] = np.log(train.Sales)
        train['log_sales_mean'] = np.log(train.sales_mean)
        train['log_sales_std'] = np.log(train.sales_std)

        # we will calculate the ratio of the sales weekend/week
        week = train[train.DayOfWeek.isin([0,1,2,3,4])][["Sales","Store"]].groupby(["Store"]).mean()
        saturday = train[train.DayOfWeek==5][["Sales","Store"]].groupby(["Store"]).mean()
        saturday_week = (saturday/week).reset_index().rename(columns={"Sales":"ratio-saturday-week"})
        train = train.merge(saturday_week , on="Store" , how="left")

        sunday = train[train.DayOfWeek==6][["Sales","Store"]].groupby(["Store"]).mean()
        sunday_week = (sunday/week).reset_index().rename(columns={"Sales":"ratio-sunday-week"})
        train = train.merge(sunday_week , on="Store" , how="left")

        # Since a lot of stores are closed on sundays, we will have to fill the nan values by 0
        train['ratio-sunday-week'].fillna(0 , inplace=True)
    
    return train


# In[11]:


## Load Data

## Load the training, test and store data 
types = {'CompetitionOpenSinceYear': np.dtype(int),
         'CompetitionOpenSinceMonth': np.dtype(int),
         'StateHoliday': np.dtype(str),
         'Promo2SinceWeek': np.dtype(int),
         'SchoolHoliday': np.dtype(int),
         'PromoInterval': np.dtype(str)}
train_data = pd.read_csv("train.csv", parse_dates=[2], dtype=types)
test_data = pd.read_csv("test.csv", parse_dates=[3], dtype=types)
store = pd.read_csv("store.csv")

## Apply preprocessing 
train,test = preprocessing(train_data,test_data,store)

## Apply feature engineering
train = feature_engineering(train)
test= feature_engineering(test, set_type='test')

 


# In[34]:


## Candidate features for training
features= np.intersect1d(train.columns,test.columns)
X_train = train[features]
Y_train = train['Sales']
X_test = test[features]


# In[ ]:


## Data is now ready for train and prediction !!

