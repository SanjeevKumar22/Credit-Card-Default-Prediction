
#Importing Libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


df_orig = pd.read_excel('/Users/priyanshutuli/Downloads/Project-Data-Set-Repository-master/Dataset/default_of_credit_card_clients.xls')

#finding rows having all zero values
df_zero_mask = df_orig == 0
feature_zero_mask = df_zero_mask.iloc[:,1:].all(axis=1)
df_clean = df_orig.loc[~feature_zero_mask,:].copy()

#Education (1 = graduate school; 2 = university; 3 = high school; 4 = others)
df_clean['EDUCATION'].replace(to_replace=[0, 5, 6], value=4, inplace=True)

#Marriage (1 = married; 2 = single; 3 = others)
df_clean['MARRIAGE'].replace(to_replace=0, value=3, inplace=True)

#finding rows with 'PAY_1' having value 'Not avialable'
missing_pay_1_mask = df_clean['PAY_1'] == 'Not available'
df_missing_pay_1 = df_clean.loc[missing_pay_1_mask,:].copy()

#lodaing the cleaned dataset
df = pd.read_csv('/Users/priyanshutuli/Downloads/Project-Data-Set-Repository-master/Dataset/cleaned_data.csv')
features_response = df.columns.tolist()
items_to_remove = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'EDUCATION_CAT', 'graduate school', 'high school', 'none',
                   'others', 'university']
features_response = [item for item in features_response if item not in items_to_remove]

X_train, X_test, y_train, y_test = train_test_split(df[features_response[:-1]].values, df['default payment next month'].values,
test_size=0.2)

k_folds = KFold(n_splits=4, shuffle=True, random_state=1)
rf = RandomForestClassifier\
(n_estimators=200, criterion='gini', max_depth=9,
min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None,
random_state=4, verbose=1, warm_start=False, class_weight=None)


pay_1_df = df.copy()
features_for_imputation = pay_1_df.columns.tolist()
items_to_remove_2 = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'EDUCATION_CAT', 'graduate school', 'high school', 'none',
                   'others', 'university', 'default payment next month', 'PAY_1']
features_for_imputation = [item for item in features_for_imputation if item not in items_to_remove_2]


X_impute_all = pay_1_df[features_for_imputation].values
y_impute_all = pay_1_df['PAY_1'].values

rf_impute = RandomForestClassifier(n_estimators=100, max_depth=12)
rf_impute.fit(X_impute_all, y_impute_all)

df_fill_pay_1_model = df_missing_pay_1.copy()
df_fill_pay_1_model['PAY_1'] = rf_impute.predict(df_fill_pay_1_model[features_for_imputation].values)
X_fill_pay_1_train, X_fill_pay_1_test, y_fill_pay_1_train, y_fill_pay_1_test = \
train_test_split(
    df_fill_pay_1_model[features_response[:-1]].values,
    df_fill_pay_1_model['default payment next month'].values,
test_size=0.2, random_state=24)

X_train_all = np.concatenate((X_train, X_fill_pay_1_train), axis=0)
y_train_all = np.concatenate((y_train, y_fill_pay_1_train), axis=0)
y_test_all = np.concatenate((y_test, y_fill_pay_1_test), axis=0)
X_test_all = np.concatenate((X_test, X_fill_pay_1_test), axis=0)


imputation_compare_cv = cross_validate(rf, X_train_all, y_train_all, scoring='roc_auc',
                                       cv=k_folds, n_jobs=-1, verbose=1,
                                       return_train_score=True, return_estimator=True,
                                       error_score='raise-deprecating')


X_fill_pay_1_train, X_fill_pay_1_test, y_fill_pay_1_train, y_fill_pay_1_test = \
train_test_split(
    df_fill_pay_1_model[features_response[:-1]].values,
    df_fill_pay_1_model['default payment next month'].values,
test_size=0.2)


rf.fit(X_train_all, y_train_all)
y_test_all_predict_proba = rf.predict_proba(X_test_all)
print(roc_auc_score(y_test_all, y_test_all_predict_proba[:,1]))
pickle.dump(rf,open('model.pkl','wb'))
