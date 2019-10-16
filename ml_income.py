

#install CATBOOST
pip install catboost

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as seabornInstance 
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score


#initializing Scaler for normalization
scaler = StandardScaler()

train_file_loc='/content/drive/My Drive/Colab Notebooks/Training_data.csv'
test_file_loc='/content/drive/My Drive/Colab Notebooks/Test_data.csv'
fileName_to_write = '/content/drive/My Drive/Colab Notebooks/Sumission_file.csv'

# to add noise on the data if dataframe has any NAN values
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


#Function to apply Target encoding on categorial data it takes train and test column along with target column to give logical mean value to strings.

def target_encode(trn_series, tst_series,target):
    min_samples_leaf=1
    smoothing=1
    noise_level=0
    temp = pd.concat([trn_series, target], axis=1)
    
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    
    # Apply average function to all target data
    prior = target.mean()
    
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    
    # Apply averages to train and test series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


#preprocessing data 
#   1. fill numerical values with mean
#   2. fill string values with FFILL method
#   3. Encode string values
 #Prepare Columns
def prepareData(dataset,dataWithoutLabels):
   
    #Train_data_preprocessing (fillna, )
    dataset=dataset.fillna(dataset.mean())
    dataset=dataset.fillna(method='ffill')
    dataset['Gender']=dataset['Gender'].replace(['unknown','0'],'UNKOWN')
    dataset['Hair Color']=dataset['Hair Color'].replace(['unknown','0'],'UNKOWN')
    
    
    
    #Test_dataset
    dataWithoutLabels=dataWithoutLabels.fillna(dataWithoutLabels.mean())
    dataWithoutLabels=dataWithoutLabels.fillna(method='ffill')
    dataWithoutLabels['Gender']=dataWithoutLabels['Gender'].replace(['unknown','0'],'UNKOWN')
    dataWithoutLabels['Hair Color']=dataset['Hair Color'].replace(['unknown','0',np.nan],'UNKOWN')
    
    #encoding of categorical data
    
    dataset['Gender'],dataWithoutLabels['Gender']=target_encode(dataset['Gender'], dataWithoutLabels['Gender'],dataset['Income in EUR'])
    dataset['Country'],dataWithoutLabels['Country']=target_encode(dataset['Country'], dataWithoutLabels['Country'],dataset['Income in EUR'])
    dataset['Profession'],dataWithoutLabels['Profession']=target_encode(dataset['Profession'], dataWithoutLabels['Profession'],dataset['Income in EUR'])
    dataset['University Degree'],dataWithoutLabels['University Degree']=target_encode(dataset['University Degree'], dataWithoutLabels['University Degree'],dataset['Income in EUR'])
    dataset['Hair Color'],dataWithoutLabels['Hair Color']=target_encode(dataset['Hair Color'], dataWithoutLabels['Hair Color'],dataset['Income in EUR'])
    
    
    return dataset,dataWithoutLabels
  
  # plot the Values on a graph
def plotActualVsPredictedIncome(y_test, y_pred):
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(25)
    df1.plot(kind='bar',figsize=(10,8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()    

def trainModel(X_train, y_train, X_test, y_test):
   
    
    #Implementing linear regression (error rate 110000+)
    #regressor = LinearRegression()  
    #regressor.fit(X_train, y_train)
    #y_pred = regressor.predict(X_test)
    
    
    #Implementing RandomForest Regressor (error rate 80000+)
    #trainingRegressor = RandomForestRegressor(n_estimators=1000)
    #trainingRegressor.fit(X_train,y_train)
    #y_pred = trainingRegressor.predict(X_test)
    
    
    
    #Implementing caatboost (error rate 60000+)
    #learning rate to get an optimal solution 
    
    trainingRegressor = CatBoostRegressor(iterations=7000,learning_rate=0.02)
    trainingRegressor.fit(X_train, y_train,use_best_model=True,verbose=True)
    y_pred=trainingRegressor.predict(X_test)
    
    
    #evaluating the error
    df = pd.DataFrame({'Test': y_test, 'Prediction': y_pred})
    print("Mean squared error: %.2f"
#       % np.sqrt(mean_squared_error(y_test, y_pred)))
    
    #ploting the values on the graph
    #plotActualVsPredictedIncome(y_test, y_pred)

    return trainingRegressor
 

def runTrainedModelOnActualData(regressor, featuresToConsider,dataWithoutLabels):
   
    #get the values for X_test    
    X = dataWithoutLabels[featuresToConsider].values
    
    #prediciting the values on the real test data
    predictedData = regressor.predict(X)
    df = pd.DataFrame({'Predicted': predictedData})

    #write the data to csv
   
    df.to_csv(fileName_to_write, sep=',',index=False)
    
    

    

 
#Reading data from CSV
dataset = pd.read_csv(Train_data_file_loc)
dataWithoutLabels = pd.read_csv(test_file_loc)

#preprocessing data on both the datasets
dataset,dataWithoutLabels = prepareData(dataset,dataWithoutLabels)

featuresToConsider = ['Year of Record', 'Age','Body Height [cm]', 'Gender','Hair Color','Wears Glasses','Country', 'Profession','University Degree']

#getting thte target data
y = dataset['Income in EUR'].values

#Normalization
dataset[featuresToConsider] = pd.DataFrame(scaler.fit_transform(dataset[featuresToConsider]), columns=featuresToConsider)
dataWithoutLabels[featuresToConsider] = pd.DataFrame(scaler.fit_transform(dataWithoutLabels[featuresToConsider]), columns=featuresToConsider)

#Features Dataset
X = dataset[featuresToConsider].values

#divide the training data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#get the trained model
regressor = trainModel(X_train, y_train, X_test, y_test)

#run on the real test data using the trained model
runTrainedModelOnActualData(regressor, featuresToConsider,dataWithoutLabels)




