from sklearn.model_selection import GridSearchCV 
from lightgbm import LGBMClassifier
import joblib
import pickle 
with open('yy.pickle' ,'rb') as f:
    x_train , y_train  =  pickle.load(f)

model_lgb = LGBMClassifier(num_class =10 ,
                           objective='multiclass',
                           metric='multi_logloss')

parameters ={
    'max_leaf': [100 , 200, 300 ,400],
    'max_depth': [2, 5 ,10 , 15, 20,  35],
    'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
    'feature_fraction': [0.6, 0.7, 0.8, 0.95],
    'bagging_fraction': [0.6, 0.7, 0.8, 0.95],
    'bagging_freq': [2, 4, 6, 8],
    'lambda_l1': [0.1, 0.4, 0.6],
    'lambda_l2': [10, 15, 35],
    'cat_smooth': [1, 10, 15, 20, 35],
    }
gs = GridSearchCV(model_lgb , 
                  param_grid =parameters , 
                  n_jobs = 20)
gs.fit(x_train , y_train)

joblib.dump(gs, 'gs_model_yy')
print('GS Done')
