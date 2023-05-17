import pandas as pd
import os
import seaborn as sns
import env
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from termcolor import colored
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from pydataset import data
from sklearn.linear_model import LassoLars
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TweedieRegressor

def check_file_exists(fn, query, url):
    """
    check if file exists in my local directory, if not, pull from sql db
    return dataframe
    """
    if os.path.isfile(fn):
        print('csv file found and loaded')
        return pd.read_csv(fn, index_col=0)
    else:
        print('creating df and exporting csv')
        df = pd.read_sql(query, url)
        df.to_csv(fn)
        return df

def get_zillow_data():
    url = env.get_connection_url('zillow')
    filename = 'zillow.csv'
    query = '''select pro.bathroomcnt, pro.bedroomcnt, pro.fips, pro.garagecarcnt, pro.yearbuilt, 
    pro.lotsizesquarefeet, pro.taxvaluedollarcnt, p2.transactiondate
    from properties_2017 pro
    join predictions_2017 p2 using (parcelid)
     where propertylandusetypeid = 261;'''
    df = check_file_exists(filename, query, url)

    return df

    
def prep_zillow(df):
    #sum_null = df.isnull().sum()
    df = df.dropna()
    df = df.drop_duplicates(keep='last')
    new_columns = {'bedroomcnt': 'bedrooms', 'bathroomcnt': 'bathrooms', 'lotsizesquarefeet': 'lotsqft', 'taxvaluedollarcnt': 'value', 'yearbuilt': 'yr built', 'fips': 'county', 'garagecarcnt': 'car garage',  'transactiondate': 'date'}
    df = df.rename(columns=new_columns)
    df['county'] = df['county'].replace({6037: 'LA', 6111: 'Ventura', 6059: 'Orange County'})
    #print(colored(f'Columns with null values: {sum_null[sum_null > 0].index}', 'red'))
    return df

def prep_z(df):
    u_columns = {'county', 'car garage', 'date'} 
    df = df.drop(u_columns, axis = 1)
    return df

def remove_outliers(df, threshold=3):
    """
    Removes outliers from the input data using the z-score method.
    Returns the filtered data without outliers.
    
    Parameters:
    -----------
    data : numpy array
        The input data.
    threshold : float
        The z-score threshold for outlier detection.
    
    Returns:
    --------
    filtered_data : numpy array
        The filtered data without outliers.
    """
    # Calculate the z-scores for each data point
    z_scores = np.abs((df - np.mean(df)) / np.std(df))
    
    # Identify the outliers based on the z-score threshold
    outliers = z_scores > threshold
    
    # Remove the outliers from the data
    filtered_data = df[~outliers]
    
    return df

def train_split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    return train, validate, test DataFrames.
    '''
    
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123)
    
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=123)

    # Take a look at your split datasets

    print(f"""
    train -> {train.shape}
    validate -> {validate.shape}
    test -> {test.shape}""")
    
    return train, validate, test

def split_data_axis(df):
    X = df.drop(['value'], axis=1)
    Y = df.value
    
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    
    return X, Y

def train_validate_test_split(X, Y):
    '''
    This function takes in a dataframe, the target variable, and a seed for reproducibility.
    It will split the data into train, validate, and test datasets.
    '''

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.3, random_state=123)
    
    print(f'''
    X_train -> {X_train.shape}'
    X_validate -> {X_validate.shape}'
    X_test -> {X_test.shape}''') 
    
    return X_train, X_validate, X_test, y_train, y_validate, y_test

def split_data(df):
    '''
    Takes in two arguments the dataframe name and the ("name" - must be in string format) to stratify  and
    return train, validate, test subset dataframes will output train, validate, and test in that order
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.25, random_state=123)
    return train, validate, test

def visualize_scaler(scaler, df, columns_to_scale, X_validate, X_test, bins=10):
    #create subplot structure
    fig, axs = plt.subplots(len(columns_to_scale), 2, figsize=(12,12))

    #copy the df for scaling
    df_scaled = df.copy()
    
    #fit and transform the df
    df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    #plot the pre-scaled data next to the post-scaled data in one row of a subplot
    for (ax1, ax2), col in zip(axs, columns_to_scale):
        ax1.hist(df[col], bins=bins)
        ax1.set(title=f'{col} before scaling', xlabel=col, ylabel='count')
        ax2.hist(df_scaled[col], bins=bins)
        ax2.set(title=f'{col} after scaling with {scaler.__class__.__name__}', xlabel=col, ylabel='count')
    plt.tight_layout()
    plt.show()
    return df_scaled, scaler.transform(X_validate[columns_to_scale]), scaler.transform(X_test[columns_to_scale])
    
def metrics_reg(y, yhat):
    """
    send in y_true, y_pred & returns RMSE, R2
    """
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    
    return rmse, r2

def eval_dist(r, p, α=0.05):
    if p > α:
        return print(f"""The data is normally distributed""")
    else:
        return print(f"""The data is NOT normally distributed""")
    
def eval_Spearman(r, p, α=0.05):
    if p < α:
        return print(f"""We reject H₀, there is a monotonic relationship.
Spearman's r: {r:2f}
P-value: {p}""")
    else:
        return print(f"""We fail to reject H₀: that there is a monotonic relationship.
Spearman's r: {r:2f}
P-value: {p}""")
    
def eval_Pearson(r, p, α=0.05):
    if p < α:
        return print(f"""We reject H₀, there is a linear relationship with a Correlation Coefficient of {r:2f}.
P-value: {p}""")
    else:
        return print(f"""We fail to reject H₀: that there is a linear relationship.
Pearson's r: {r:2f}
P-value: {p}""")

    
import scipy.stats as stats
import pandas as pd
import os
import numpy as np

# Data viz:
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn stuff:
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import  mean_squared_error
from math import sqrt


import env


#This function automates the process for modeling and evaluating regression models
def auto_regress( y_train, train_df, x_train_scaled, x_validate_scaled, y_validate, x_test_scaled):

    def regression_errors(y, yhat):
        '''
        Returns the following values:
        mean squared error (MSE)
        root mean squared error (RMSE)
        '''
        #import
        
        #calculate r2
        r2 = r2_score(y, yhat)
        #calculate MSE
        MSE = mean_squared_error(y, yhat)
        #calculate RMSE
        RMSE = sqrt(MSE)
        
        return RMSE, r2

        
        
        
    
    def model_all(y_train, train_df, x_train_scaled, x_validate_scaled,y_validate,x_test_scaled):
        baseline = y_train.mean()
        
        #calculate baseline
        baseline_array = np.repeat(baseline, len(train_df))

        RMSE, r2 = regression_errors(y_train, baseline_array)
        metric_df = pd.DataFrame(data=[{
        'model': 'mean_baseline',   
        'RMSE': RMSE,
        'r^2': r2}])
        
        #OLS_1 and RFE
        #intial model
        Lr1 = LinearRegression()
        #make the model
        rfe = RFE(Lr1, n_features_to_select=1)
        #fit the model
        rfe.fit(x_train_scaled, y_train)
        #use it on train
        x_train_scaled_rfe = rfe.transform(x_train_scaled)
        #use it on validate
        print(x_train_scaled)
        x_validate_scaled_rfe = rfe.transform(x_validate_scaled)

        rfe_ranking = pd.DataFrame({'rfe_ranking': rfe.ranking_}, index=x_train_scaled.columns)
        rfe_ranking.sort_values(by=['rfe_ranking'], ascending=True).head(1)

        #build the model from top feature
        #fit the model
        Lr1.fit(x_train_scaled_rfe, y_train)
        #predict
        pred_Lr1 = Lr1.predict(x_train_scaled_rfe)
        pred_val_Lr1 = Lr1.predict(x_validate_scaled_rfe)

        #evaluate Lr1
        #evaluate on train
        regression_errors(y_train, pred_Lr1)
        #evaluate on validate
        rmse, r2 = regression_errors(y_validate, pred_val_Lr1)

        #add to metric_df
        metric_df.loc[1] = ['ols_1', rmse, r2]
       
        #multiple Regression with OLS
        #make the model
        Lr2 = LinearRegression(normalize=True)
        #fit the model
        Lr2.fit(x_train_scaled, y_train)
        #predict
        pred_Lr2 = Lr2.predict(x_train_scaled)
        #predict validate
        pred_val_Lr2 = Lr2.predict(x_validate_scaled)

        #evaluate Lr2
        #evaluate on train
        regression_errors(y_train, pred_Lr2)
        #evaluate on validate
        rmse, r2 = regression_errors(y_validate, pred_val_Lr2)

        #add to metric_df
        metric_df.loc[2] = ['ols_2', rmse, r2]

        #LassoLars
        #make the model
        lars = LassoLars(alpha=4)
        #fit the model
        lars.fit(x_train_scaled, y_train)
        #predict
        pred_lars = lars.predict(x_train_scaled)
        #predict validate
        pred_val_lars = lars.predict(x_validate_scaled)

        #evaluate lars
        #train
        rmse, r2= regression_errors(y_validate, pred_val_lars)

        #add to metric_df
        metric_df.loc[3] = ['lars', rmse, r2]

        #polynomial regression
        #make polynomial features to get a new set of features
        pf = PolynomialFeatures(degree=3)
        #fit and transform x_train_scaled
        x_train_scaled_pf = pf.fit_transform(x_train_scaled)
        #transform x_validate_scaled and x_test_scaled
        x_validate_scaled_pf = pf.transform(x_validate_scaled)
        x_test_scaled_pf = pf.transform(x_test_scaled)

        #fit to linear regression model
        #make the model
        pr = LinearRegression()
        #fit the model
        pr.fit(x_train_scaled_pf, y_train)
        #predict
        pred_pr = pr.predict(x_train_scaled_pf)
        #predict validate
        pred_val_pr = pr.predict(x_validate_scaled_pf)

        #evaluate pr
        regression_errors(y_train, pred_pr)
        rmse, r2 = regression_errors(y_validate, pred_val_pr)

        metric_df.loc[4] = ['poly', rmse, r2]

        #tweedie regression
        #make the model
        glm = TweedieRegressor(power=1, alpha=0)
        #fit the model
        glm.fit(x_train_scaled, y_train)
        #predict
        pred_glm = glm.predict(x_train_scaled)
        #predict validate
        pred_val_glm = glm.predict(x_validate_scaled)

        #evaluate glm
        regression_errors(y_train, pred_glm)
        rmse, r2 = regression_errors(y_validate, pred_val_glm)

        metric_df.loc[5] = ['glm', rmse, r2]

        print("\n")
        print("The best model is the", metric_df.loc[metric_df['RMSE'].idxmin()][0], "model\n")
        
        #plot actuals vs predicted
        plt.figure(figsize=(16,8))
        plt.plot(y_validate, y_validate, color='gray', label='Perfect Model')
        
        
        plt.scatter(y_validate, pred_val_lars, color='blue', alpha=.5, label='Model 1: LassoLars')
        plt.scatter(y_validate, pred_val_pr, color='green', alpha=.5, label='Model 2: PolynomialRegression')
        plt.scatter(y_validate, pred_val_glm, color='red', alpha=.5, label='Model 3: TweedieRegressor')
        #plot the baseline line
        plt.legend()
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs. Predicted")
        plt.show()
        print(metric_df)

    ''' 
       #plot residuals
        plt.figure(figsize=(16,8))
        plt.axhline(label="No Error")
        plt.scatter(y_validate, pred_val_lars - y_validate, alpha=.5, color="blue", s=100, label="Model 1: LassoLars")
        plt.scatter(y_validate, pred_val_pr - y_validate, alpha=.5, color="green", s=100, label="Model 2: PolynomialRegression")
        plt.scatter(y_validate, pred_val_glm - y_validate, alpha=.5, color="red", s=100, label="Model 3: TweedieRegressor")
        plt.legend()
        plt.xlabel("Actual")
        plt.ylabel("Residual/Error: Predicted - Actual")
        plt.title("Do the size of errors change as the actual value changes?")
        plt.show()'''

        
    model_all(y_train, train_df, x_train_scaled, x_validate_scaled, y_validate, x_test_scaled)
    
def lot_sf_plot_lmplot(train):
    """
    Creates an lmplot of finished_sf vs. taxvalue for a sample of 5000 rows from the training set.
    """
    # Create a sample of 5000 rows from the training set.
    train_sample = train.sample(n=5000, random_state=42)
    # Create lmplot.
    sns.lmplot(x='lotsqft', y= 'value', data=train_sample)
plt.show()

def bed_plot_lmplot(train):
    """
    Creates an lmplot of bedrooms vs. value for a sample of 5000 rows from the training set.
    """
    # Create a sample of 5000 rows from the training set.
    train_sample = train.sample(n=5000, random_state=42)
    # Create lmplot.
    sns.lmplot(x= 'bedrooms', y= 'value', data=train_sample)
plt.show()

def bath_plot_lmplot(train):
    """
    Creates an lmplot of bathrooms vs. value for a sample of 5000 rows from the training set.
    """
    # Create a sample of 5000 rows from the training set.
    train_sample = train.sample(n=5000, random_state=42)
    # Create lmplot.
    sns.lmplot(x='bathrooms', y= 'value', data=train_sample)
plt.show()

def metrics_reg(y, yhat):
    """
    send in y_true, y_pred & returns RMSE, R2
    """
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return rmse, r2