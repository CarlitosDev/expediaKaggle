'''
    Solution for the Kaggle challenge

    Carlos Aguilar - 25th May

'''

import pandas as pd
import os
import numpy as np
import gc
from datetime import datetime, timedelta, date
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import itertools


csvFolder   = '/Users/carlos.aguilar/.kaggle/competitions/hotelsales2'

# - - - - - - - - - -
# Read the test set in the first place (short on time)
csvFile  = 'test.csv'
csvPath  = os.path.join(csvFolder, csvFile);
dfTest   = pd.read_csv(csvPath)
[r,c]    = dfTest.shape
varNames = dfTest.keys().tolist()
print('File {} has got {} rows and {} columns ({})'.format(csvFile, r, c, varNames))


# - - - - - - - - - -
# What's in the HOTELS data?
csvFile  = 'hotels.csv'
csvPath  = os.path.join(csvFolder, csvFile);
dfHotels = pd.read_csv(csvPath)
[r,c]    = dfHotels.shape
varNames = dfHotels.keys().tolist()
print('File {} has got {} rows and {} columns ({})'.format(csvFile, r, c, varNames))
dfHotels.iloc[32]

# Let's focus on the hotels to forecast for
idHotelsTestSet = dfTest.hotel.unique().tolist()
dfHotelsTest    = dfHotels[dfHotels.hotelid.isin(idHotelsTestSet)]
dfHotelsTest.iloc[43]

# Check the countries, etc
varsToEval = ['city', 'country', 'airport', 'category']
unqVals    = []
for iVar in varsToEval:
    unqVals.append(dfHotelsTest[iVar].unique())
    print('Number of unique {} is {}'.format(iVar, len(unqVals[-1])))

# So...129 countries to predict. Just that...
# Let's save the unique values
uniqueValsDict = dict(zip(varsToEval, unqVals))




# - - - - - - - - - -
# Read the TRAINING set
csvFile  = 'hotel_sales_daily.csv'
csvPath  = os.path.join(csvFolder, csvFile);
dfTrain  = pd.read_csv(csvPath, parse_dates=['trans_date']).set_index('trans_date')
[r,c]    = dfTrain.shape
varNames = dfTrain.keys().tolist()
print('File {} has got {} rows and {} columns ({})'.format(csvFile, r, c, varNames))

minDate = dfTrain.index.min();
maxDate = dfTrain.index.max();
print('Date range from {} to {}'.format(minDate, maxDate))

uniqueHotels_dfTrain = dfTrain['hotel'].unique().tolist();
print('Number of hotels in the training set {}'.format(len(uniqueHotels_dfTrain)))

# Let's reduce the 185588 (???) to the ones that are in the same countries as the
# ones in the test set.
idxTrainHotels   = dfHotels.hotelid.isin(uniqueHotels_dfTrain)
dfHotelsTraining = dfHotels[idxTrainHotels]
idxCountry       = dfHotelsTraining.country.isin(list(uniqueValsDict['country']))
dfHotelsTraining = dfHotelsTraining[idxCountry]

idxRedux         = dfTrain.hotel.isin(dfHotelsTraining.hotelid.unique().tolist())
dfTrain_reduced  = dfTrain[idxRedux]
# free some memory
del dfTrain
# Do the log of the sales to resemble a bit more of a Gaussian
dfTrain_reduced['log_sales'] = np.log1p(dfTrain_reduced.sales)
maxLogSales = dfTrain_reduced['log_sales'].max()
minLogSales = dfTrain_reduced['log_sales'].min()

# Pick the variables that will go into the model
varsModels = ['city', 'country', 
'rate1', 'rate2', 'rate3', 'capacity', 
'reviewrating', 'reviewcnt', 'radius500h', 'hotelid']
dfHotelsMini = dfHotelsTraining[varsModels].set_index('hotelid')



# The solution is the sales from 1-10 Dec 2k17 per hotel, so from a Friday to a Sunday.
# Let's divide the historical sales into batches covering the same period

firstDay    = datetime(2017, 9, 8);
periodStart = pd.date_range(start=firstDay, freq = '14D', periods=6)
periodEnd   = periodStart + timedelta(days=9)

periodsOfSales = []
for iPeriod in range(0,6):
    thisPeriod = dfTrain_reduced.loc[pd.date_range(periodStart[iPeriod], periodEnd[iPeriod])][['hotel', 'log_sales']]
    thisPeriod = thisPeriod.groupby('hotel').agg({'sum', 'count', 'mean', 'median'})
    thisPeriodExt = pd.merge(thisPeriod, dfHotelsMini, 
        how='inner', right_index=True, left_index=True, suffixes=('', ''), copy=True);
    thisPeriodExt['weight'] = (iPeriod + 15)/30
    periodsOfSales.append(thisPeriodExt)
    
# assemble as a DF
train_set = pd.concat(periodsOfSales, axis=0)
train_set.shape

#train_set.rename(column={('log_sales', 'sum'):'manolo'}, inplace=True)
train_set['log_sales'] = train_set[('log_sales', 'sum')]
train_set['count']     = train_set[('log_sales', 'count')]

# Set the input variables and the label
targetVariable  =  'log_sales'
categoricalVars = ['city', 'country']
numericalVars   = ['rate1', 'rate2', 'rate3',
 'capacity', 'reviewrating', 'reviewcnt', 'radius500h', 'weight']


# prepare the categorical variables
leVarNames   = []
encodersList = []
for varName in categoricalVars:
    currentVarName = varName + 'CAT'
    leVarNames.append(currentVarName)
    currentLE = LabelEncoder()
    train_set[currentVarName] = currentLE.fit_transform(train_set[varName].astype(str))
    encodersList.append(currentLE)

leSolver  = dict(zip(categoricalVars, encodersList))
inputVars = list(itertools.chain.from_iterable([leVarNames, numericalVars]))

# train and validation
X_train, X_valid, y_train, y_valid = train_test_split(train_set[inputVars], 
    train_set[targetVariable], 
    test_size=0.20, random_state=42)

w_train = X_train[inputVars[-1]]


# Prepare LGBM datasets (Consider scaling the features)
# LightGBM can use categorical features as input directly. It doesn’t need 
# to convert to one-hot coding, and is much faster than one-hot coding (about 8x speed-up).
# You should convert your categorical features to int type before you construct Dataset.

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'l2',
    'num_leaves': 31,
    'learning_rate': 0.25,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'nthread': 4,
    'metric_freq': 1,
    'max_bin': 255,
    'early_stopping': 50
}

number_of_trees = 350;

train_data = lgb.Dataset(X_train[inputVars[:-1]],
    label=y_train,
    categorical_feature=leVarNames,
    weight=X_train[inputVars[-1]],
    free_raw_data=False)

valid_data = lgb.Dataset(X_valid[inputVars[:-1]],
    label=y_valid,
    categorical_feature=leVarNames,
    weight=X_valid[inputVars[-1]],
    reference=train_data,
    free_raw_data= False);

lgb_model = lgb.train(params, 
            train_data,
            valid_sets=[train_data, valid_data],
            verbose_eval=25,
            num_boost_round=number_of_trees)


# Prepare the data for testing
test_set = dfHotelsTest

# Apply the encoders
for varName in categoricalVars:
    currentVarName = varName + 'CAT'
    currentLE = leSolver[varName]
    test_set[currentVarName] = currentLE.fit_transform(test_set[varName].astype(str))


# Predict
X_test = test_set[inputVars[:-1]]
y_hat = lgb_model.predict(X_test, 
    num_iteration=lgb_model.best_iteration or number_of_trees)

# clip the predictions as the DT has gone a bit crazy
y_hat_clipped = np.clip(y_hat, minLogSales*0.7, maxLogSales*0.65)
dfHotelsTest['sales'] = np.expm1(y_hat_clipped)

coldStartForecast = dfHotelsTest[['hotelid', 'sales']]
coldStartForecast.rename(columns={'hotelid':'hotel'}, inplace=True)
csvPath = os.path.join('/Users/carlos.aguilar/Documents/Kaggle Competition/Expedia', 'carlos_aguilar_.csv');
coldStartForecast.to_csv(csvPath, float_format='%.4f', index=None)













