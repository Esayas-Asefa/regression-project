import pandas as pd
import os
import seaborn as sns
import env
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from termcolor import colored
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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
    new_columns = {'bedroomcnt': 'bedrooms', 'bathroomcnt': 'bathrooms', 'calculatedfinishedsquarefeet': 'sqft', 'taxvaluedollarcnt': 'value', 'yearbuilt': 'yr built', 'fips': 'county', 'garagecarcnt': 'car garage', 'lotsizesquarefeet': 'sq ft', 'transactiondate': 'date'}
    df = df.rename(columns=new_columns)
    df['county'] = df['county'].replace({6037: 'LA', 6111: 'Ventura', 6059: 'Orange County'})
    #print(colored(f'Columns with null values: {sum_null[sum_null > 0].index}', 'red'))
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

def train_validate_test_split(df):
    '''
    This function takes in a dataframe, the target variable, and a seed for reproducibility.
    It will split the data into train, validate, and test datasets.
    '''

    X_train, X_test, y_train, y_test = train_test_split(X_all, Y, test_size=0.2, random_state=123)
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

def visualize_scaler(scaler, df, columns_to_scale, bins=10):
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
