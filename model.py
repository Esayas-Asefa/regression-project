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
    
acq.auto_regress(y_train, X_train, X_train, X_test, y_test, X_test)

acq.auto_regress(y_train, X_train, X_train, X_validate, y_validate, X_test)
