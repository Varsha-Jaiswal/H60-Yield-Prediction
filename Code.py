#Libraries Used
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation,metrics
from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
le = LabelEncoder()
from matplotlib.pylab import rcParams
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn import preprocessing
%matplotlib inline
rcParams['figure.figsize'] = 12,4

def linear_regression(dtrain,dtest,predictors,target,Id, algo = 'linear',print_coef=False,test_size=0.2,cv=True,feature_importance=True,random_state=100,alpha =0.05,l1_ratio = 0.5):
    if algo == 'linear':
        model = linear_model.LinearRegression()
    elif algo == 'ridge':
        model = linear_model.Ridge(alpha = alpha, normalize =True)
    elif algo == 'lasso':
        model = linear_model.Lasso(alpha = alpha, normalize =True)
    elif algo == 'elasticnet':
        model = linear_model.ElasticNet(alpha = alpha, l1_ratio = l1_ratio, normalize = True)
    if cv:
        x_train,x_cv,y_train,y_cv = train_test_split(dtrain.loc[:,predictors],dtrain.loc[:,target],test_size=test_size,random_state=random_state)
        model.fit(x_train,y_train)
        print(metrics.r2_score(y_train,model.predict(x_train.loc[:,predictors])))
        print(metrics.r2_score(y_cv,model.predict(x_cv.loc[:,predictors])))
    else:
        model.fit(dtrain.loc[:,predictors],dtrain[target])
        print(model.score(dtrain.loc[:,predictors],dtrain[target]))
    if print_coef:
        coef = Series(model.coef_,predictors).sort_values()
        coef.plot(kind='bar',title='Modal Coefficients')
    if feature_importance:
        features = pd.DataFrame()
        features['feature'] = predictors
        features['importance'] = np.absolute(model.coef_)
        features.sort_values(by=['importance'], ascending=True, inplace=True)
        features.set_index('feature', inplace=True)
        features.plot(kind='barh', figsize=(25, 25))
    predicted = model.predict(dtest.loc[:,predictors])   
    return predicted

	
train = pd.read_csv("/home/ubuntu/Hackathons/Yield Prediction/train.csv")
test = pd.read_csv("/home/ubuntu/Hackathons/Yield Prediction/test.csv")
target = 'Production'
ID = 'Id'
targets = train[target]
train.drop([target],1,inplace = True)
combined = train.append(test)

var_to_encode = ['State_Name','Season','District_Name','Crop']
for col in var_to_encode:
    combined[col] = le.fit_transform(combined[col])
combined = pd.get_dummies(combined, columns=var_to_encode)
x = combined[['Area']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
combined_new = combined
combined_new['Area'] = x_scaled 
train_new = combined_new.iloc[:217510]
test_new = combined_new.iloc[217510:]
train_ids = train_new[ID]
test_ids = test_new[ID]
train_new.drop([ID],1,inplace = True)
test_new.drop([ID],1,inplace = True)
pca = PCA(.95)
pca.fit(train_new)
train_new2 = pca.transform(train_new)
test_new2 = pca.transform(test_new)
train_new3 = pd.DataFrame(train_new2)
test_new3 = pd.DataFrame(test_new2)
train_new4 = train_new
test_new4 = test_new
train_new4[ID] = train_ids
test_new4[ID] = test_ids
train_new4[target] = targets

predictors = [x for x in train_new4.columns if x not in [target,ID]]
xgb = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.08, gamma=0, subsample=0.75,colsample_bytree=1, max_depth=7)
x_train,x_cv,y_train,y_cv = train_test_split(train_new4.loc[:,predictors],train_new4.loc[:,target],test_size=0.2)
xgb.fit(x_train,y_train)
print(metrics.r2_score(y_train,xgb.predict(x_train.loc[:,predictors])))
print(metrics.r2_score(y_cv,xgb.predict(x_cv.loc[:,predictors])))
predicted_xgb = xgb.predict(test_new4.loc[:,predictors]) 
print(len(predicted_xgb))
for i in range(len(predicted_xgb)):
    if(predicted_xgb[i] <0):
        predicted_xgb[i] = 0
     
predicted = pd.DataFrame()
predicted['Id'] = test[ID]
predicted['Yield'] = predicted_xgb
predicted.to_csv("/home/ubuntu/Hackathons/Yield Prediction/submission.csv",index = False)  	 
