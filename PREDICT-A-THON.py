import pandas as pd
import numpy as np
import seaborn as sns
from imblearn.under_sampling import NearMiss
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import precision_score,recall_score,log_loss,accuracy_score

data=pd.read_csv("train.csv")

cols_utilized=['price', 'basePrice','reward','premiumProduct','couponUsed']

data_utilized= data[cols_utilized]
data_utilized.skew(axis=1,skipna = True)

mask_3=data_utilized['basePrice']<15
data_utilized_mask_1=data_utilized[mask_3]
mask_4=data_utilized_mask_1['price']<30
data_utilized_mask_2=data_utilized_mask_1[mask_4]
mask_5=data_utilized_mask_2['reward']<4
data_utilized_mask_3=data_utilized_mask_2[mask_5]
processed_data=data_utilized_mask_3.copy(deep=True)

print("Number of records : ",processed_data.shape[0])
print("Number of attributes : ",processed_data.shape[1])
print("Percentage of outliers dropped : ",np.round((data_utilized.shape[0]-processed_data.shape[0])/len(data_utilized)*100,2))

processed_data.skew(axis=1,skipna = True)

processed_data['price_log']=np.log(processed_data['price'])
processed_data['basePrice_log']=np.sqrt(processed_data['basePrice'])
processed_data_uti=processed_data[['price_log','basePrice_log','reward','premiumProduct','couponUsed']]
processed_data_uti.columns=['price', 'basePrice','reward','premiumProduct','couponUsed']    
processed_data_uti.skew(axis=0)
X=processed_data_uti.drop(['couponUsed'],axis=1)
y=processed_data_uti['couponUsed']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

undersmaple=NearMiss(version=3, n_neighbors_ver3=3)

X_under,y_under=undersmaple.fit_resample(X,y)
X_train_under, X_test_under, y_train_under, y_test_under = train_test_split(X_under,y_under, test_size=0.20, random_state=42)
X_train_under.shape, X_test_under.shape, y_train_under.shape, y_test_under.shape

LR_model=LogisticRegression()
LR_model.fit(X_train,y_train)
y_pred_LR_base=LR_model.predict(X_test)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb_based=gnb.predict(X_test)

RF_base=RandomForestClassifier()
RF_base.fit(X_train,y_train)
y_pred_rf_base=RF_base.predict(X_test)

print("Log Loss score : ",np.round(log_loss(y_test,y_pred_LR_base),2))
print("Precision score",precision_score(y_test,y_pred_LR_base,zero_division=0))
print("Recall score",recall_score(y_test,y_pred_LR_base))
print("Accuracy score",np.round(accuracy_score(y_test,y_pred_LR_base),2))

print("Log Loss score : ",np.round(log_loss(y_test,y_pred_gnb_based),2))
print("Precision score",precision_score(y_test,y_pred_gnb_based,zero_division=0))
print("Recall score",recall_score(y_test,y_pred_gnb_based))
print("Accuracy score",np.round(accuracy_score(y_test,y_pred_gnb_based),2))

print("Log Loss score : ",np.round(log_loss(y_test,y_pred_rf_base),2))
print("Precision score",np.round(precision_score(y_test,y_pred_rf_base,zero_division=0),2))
print("Recall score",np.round(recall_score(y_test,y_pred_rf_base),2))
print("Accuracy score",np.round(accuracy_score(y_test,y_pred_rf_base),2))

LR_model_tunned=LogisticRegression(solver='sag',
                            penalty=None,
                            max_iter=1000,
                            multi_class='ovr',
                            warm_start=True)
LR_model_tunned.fit(X_train,y_train)
y_pred_LR_tunned=LR_model_tunned.predict(X_test)


gnb_tunned = GaussianNB(var_smoothing=0.06)
gnb_tunned.fit(X_train, y_train)
y_pred_gnb_tunned=gnb_tunned.predict(X_test)



RF_tuned=RandomForestClassifier(n_estimators= 1600,
 min_samples_split=5,
 min_samples_leaf= 8,
 max_features='sqrt',
 max_depth=20,
 bootstrap= True,
 oob_score=True)
RF_tuned.fit(X_train,y_train)
y_pred_RF_tuned=RF_tuned.predict(X_test)
y_pred_RF_tuned_clipped=np.clip(y_pred_RF_tuned,0.02,0.98)

param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}
 
# Instantiating Decision Tree classifier
tree = DecisionTreeClassifier()
 
# Instantiating RandomizedSearchCV object
RSC_tuned = RandomizedSearchCV(tree, param_dist, cv = 5)
 
RSC_tuned.fit(X_train, y_train)
y_pred_RSC_tuned=RSC_tuned.predict(X_test)


print("Log Loss score : ",np.round(log_loss(y_test,y_pred_LR_tunned),2))
print("Precision score",precision_score(y_test,y_pred_LR_tunned,zero_division=0))
print("Recall score",recall_score(y_test,y_pred_LR_tunned))
print("Accuracy score",np.round(accuracy_score(y_test,y_pred_LR_tunned),2))

print("Log Loss score : ",np.round(log_loss(y_test,y_pred_gnb_tunned),2))
print("Precision score",precision_score(y_test,y_pred_gnb_tunned,zero_division=0))
print("Recall score",recall_score(y_test,y_pred_gnb_tunned))
print("Accuracy score",np.round(accuracy_score(y_test,y_pred_gnb_tunned),2))

print("Log Loss score : ",np.round(log_loss(y_test,y_pred_RF_tuned),2))
print("Precision score",np.round(precision_score(y_test,y_pred_RF_tuned,zero_division=0),2))
print("Recall score",np.round(recall_score(y_test,y_pred_RF_tuned),2))
print("Accuracy score",np.round(accuracy_score(y_test,y_pred_RF_tuned),2))

print("Log Loss score : ",np.round(log_loss(y_test,y_pred_RSC_tuned),2))
print("Precision score",np.round(precision_score(y_test,y_pred_RSC_tuned,zero_division=0),2))
print("Recall score",np.round(recall_score(y_test,y_pred_RSC_tuned),2))
print("Accuracy score",np.round(accuracy_score(y_test,y_pred_RSC_tuned),2))

RF_tuned_2_under=RandomForestClassifier(n_estimators= 1600,
 min_samples_split=5,
 min_samples_leaf= 8,
 max_features='sqrt',
 max_depth=None,
 bootstrap= True,
 oob_score=True)
RF_tuned_2_under.fit(X_train_under,y_train_under)
y_pred_RF_under=RF_tuned_2_under.predict(X_test_under)
y_pred_RF_under_clipped=np.clip(y_pred_RF_under,0.02,0.98)

print("Precision score for under sample",np.round(precision_score(y_test_under,y_pred_RF_under),2))
print("Recall score for under sample",np.round(recall_score(y_test_under,y_pred_RF_under),2))
print("Accuracy score for under sample",np.round(accuracy_score(y_test_under,y_pred_RF_under),2))

LR_base_ll=np.round(log_loss(y_test,y_pred_LR_base),2)
LR_base_precision=precision_score(y_test,y_pred_LR_base,zero_division=0)
LR_base_recall=recall_score(y_test,y_pred_LR_base)
LR_base_acc=np.round(accuracy_score(y_test,y_pred_LR_base),2)
#Storing scores for base GaussianNB model
GNB_base_LL=np.round(log_loss(y_test,y_pred_gnb_based),2)
GNB_base_precision=precision_score(y_test,y_pred_gnb_based,zero_division=0)
GNB_base_recall=recall_score(y_test,y_pred_gnb_based)
GNB_base_acc=np.round(accuracy_score(y_test,y_pred_gnb_based),2)
#Storing scores for base RF model
RF_base_LL=np.round(log_loss(y_test,y_pred_rf_base),2)
RF_base_precision=np.round(precision_score(y_test,y_pred_rf_base,zero_division=0),2)
RF_base_recall=np.round(recall_score(y_test,y_pred_rf_base),2)
RF_base_acc=np.round(accuracy_score(y_test,y_pred_rf_base),2)
#Storing scores for tuned LR model
LR_tunned_ll=np.round(log_loss(y_test,y_pred_LR_tunned),2)
LR_tunned_precision=precision_score(y_test,y_pred_LR_tunned,zero_division=0)
LR_tunned_recall=recall_score(y_test,y_pred_LR_tunned)
LR_tunned_acc=np.round(accuracy_score(y_test,y_pred_LR_tunned),2)
#Storing scores for tunned GaussianNB model
GNB_tunned_LL=np.round(log_loss(y_test,y_pred_gnb_tunned),2)
GNB_tunned_precision=precision_score(y_test,y_pred_gnb_tunned,zero_division=0)
GNB_tunned_recall=recall_score(y_test,y_pred_gnb_tunned)
GNB_tunned_acc=np.round(accuracy_score(y_test,y_pred_gnb_tunned),2)
#Storing scores for tunned RF model
RF_tunned_LL=np.round(log_loss(y_test,y_pred_RF_tuned),2)
RF_tunned_precision=np.round(precision_score(y_test,y_pred_RF_tuned,zero_division=0),2)
RF_tunned_recall=np.round(recall_score(y_test,y_pred_RF_tuned),2)
RF_tunned_acc=np.round(accuracy_score(y_test,y_pred_RF_tuned),2)
#Storing scores for final model(RF) trained on undersampled data
final_model_LL=np.round(log_loss(y_test_under,y_pred_RF_under),2)
final_model_precision=np.round(precision_score(y_test_under,y_pred_RF_under),2)
final_model_recall=np.round(recall_score(y_test_under,y_pred_RF_under),2)
final_model_acc=np.round(accuracy_score(y_test_under,y_pred_RF_under),2)

base_scores=[LR_base_ll,LR_base_precision,LR_base_recall,LR_base_acc,
             GNB_base_LL,GNB_base_precision,GNB_base_recall,GNB_base_acc,
             RF_base_LL,RF_base_precision,RF_base_recall,RF_base_acc]
tunned_scores=[LR_tunned_ll,LR_tunned_precision,LR_tunned_recall,LR_tunned_acc,
             GNB_tunned_LL,GNB_tunned_precision,GNB_tunned_recall,GNB_tunned_acc,
             RF_tunned_LL,RF_tunned_precision,RF_tunned_recall,RF_tunned_acc]
finalmodel_scores=[final_model_LL,final_model_precision,final_model_recall,final_model_acc]


scores={'Base':{'Logistic Regression':{'Log Loss':LR_base_ll,'Precision':LR_base_precision,
                    'Recall':LR_base_recall,'Accuracy':LR_base_acc},
                    'GaussianNB':{'Log Loss':GNB_base_LL,'Precision':GNB_base_precision,
                    'Recall':GNB_base_recall,'Accuracy':GNB_base_acc},
                    'Random Forest Classifier':{'Log Loss':RF_base_LL,'Precision':RF_base_precision,
                    'Recall':RF_base_recall,'Accuracy':RF_base_acc}},
          'Tunned':{'Logistic Regression':{'Log Loss':LR_tunned_ll,'Precision':LR_tunned_precision,
                    'Recall':LR_tunned_recall,'Accuracy':LR_tunned_acc},
                    'GaussianNB':{'Log Loss':GNB_tunned_LL,'Precision':GNB_tunned_precision,
                    'Recall':GNB_tunned_recall,'Accuracy':GNB_tunned_acc},
                    'Random Forest Classifier':{'Log Loss':RF_tunned_LL,'Precision':RF_tunned_precision,
                    'Recall':RF_tunned_recall,'Accuracy':RF_tunned_acc},
                    },
          'Final':{'Random Forest':{'Log Loss':final_model_LL,'Precision':final_model_precision,
                    'Recall':final_model_recall,'Accuracy':final_model_acc}}   
            }

Final_model_scores=pd.DataFrame.from_dict(data=scores.get('Final'))
base_model_scores=pd.DataFrame.from_dict(data=scores.get('Base'))
tunned_model_scores=pd.DataFrame.from_dict(data=scores.get('Tunned'))


data_train=pd.read_csv("train.csv")
data_train=data_train[['price', 'basePrice','reward','premiumProduct','couponUsed']]
data_test=pd.read_csv("test.csv")
data_test=data_test[['price', 'basePrice','reward','premiumProduct']]

X_train_1=data_train.drop(['couponUsed'],axis=1)
y_train_1=data_train['couponUsed']

RF_tuned_2_under=RandomForestClassifier(n_estimators=1600,
 min_samples_split=5,
 min_samples_leaf= 8,
 max_features='sqrt',
 max_depth=20,
 bootstrap= True,
 oob_score=True)
RF_tuned_2_under.fit(X_train_1,y_train_1)
y_pred_RF_under=RF_tuned_2_under.predict(data_test)
y_pred_RF_under_clipped=np.clip(y_pred_RF_under,0.20,0.80)



submission_data_1=pd.DataFrame(y_pred_RF_under,columns=['couponUsed'])

submission_data_1['couponUsed']=RF_tuned_2_under.predict(data_test).clip(0.17,0.83)

submission_data_1.to_csv("SUBMISSION.csv",index=False)

# submission_data=pd.DataFrame(y_pred_RF_under,columns=['couponUsed'])

# submission_data.to_csv("submission567.csv",index=False)


