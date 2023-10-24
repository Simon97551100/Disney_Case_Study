
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,mean_absolute_error, mean_squared_error,r2_score
from sklearn.metrics import confusion_matrix, roc_curve, auc,roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2,f_classif,f_regression
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from PIL import Image


# Choose Disney Theme

Theme_Selection = st.selectbox('Which Disney Theme ?',['Cars','Bambi','Peter Pan'])
if Theme_Selection == 'Cars':
    image = Image.open('RFLMIntro.jpeg')
if Theme_Selection == 'Bambi':
    image = Image.open('RFBAintro.jpeg')
if Theme_Selection == 'Peter Pan':
    image = Image.open('RFPPintro.jpeg')

st.image(image)



st.title('Random Forest Classification/Regresssion Model')

# Upload File

Upload = st.file_uploader('Upload CSV File Here', type=None, accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
df = pd.read_csv(Upload)
st.write('Raw Dataframe')
st.write(df)

# Clean Data 

dropped_columns = st.multiselect(
    'Which Column Do You Want To Drop ?',
     list(df.columns.values))
df = df.drop(columns=dropped_columns)
dropped_rows = st.number_input("How Many Rows Do You Want To Keep ?",0,500000)
if dropped_rows != 0:
    df.drop(df.index[dropped_rows:], inplace=True)
One_Hot_List = []
for (columnName, columnData) in df.iteritems():
    if df[columnName].dtypes not in [int,float]:
        One_Hot_List.append(columnName)
df = pd.get_dummies(df, columns=One_Hot_List)
st.write('Cleansed Dataframe')
df

CleanUp_Input = st.selectbox(
    'Dataframe Cleaning Method',
     ['Fill Nan With 0','Remove Rows With Nan','Replace Nan Values With Average'])
if CleanUp_Input == 'Fill Nan With 0':
    df.fillna(0, inplace=True)
if CleanUp_Input == 'Remove Rows With Nan':
    df.dropna()
if CleanUp_Input == 'Replace Nan Values With Average':
    df = df.apply(lambda x: x.fillna(x.mean()), axis=0)

if Theme_Selection == 'Cars':
    image = Image.open('RFLMCL.jpeg')
if Theme_Selection == 'Bambi':
    image = Image.open('RFBACL.jpeg')
if Theme_Selection == 'Peter Pan':
    image = Image.open('RFPPCL.jpeg')

st.image(image)

# Pick Between Classification

ML_Task = st.selectbox(
    'What type Of Machine Learning Task ?',
     ['Classification','Regression'])

y_prediction = st.selectbox(
    'Which Column Do You Want To Predict ?',
     list(df.columns.values))

X = df.drop(columns=[y_prediction])
y = df[y_prediction]

# Set Classification Paramters

if ML_Task == 'Classification':
    Criterion = 'gini'
    Max_depth = None
    Min_samples_split = 2
    Min_samples_leaf = 1
    Min_weight_fraction_leaf = 0.0
    Max_features = 'sqrt'
    Max_leaf_nodes = None
    Min_impurity_decrease = 0.0
    Bootstrap = True
    Random_state = 42
    Max_samples= None 
    Test_size = 0.2
    Parameter_Select = st.multiselect('Select Paramters To Change', ['criterion','max_depth','min_samples_split','min_samples_leaf','min_weight_fraction_leaf','max_features','max_leaf_nodes','min_impurity_decrease','bootstrap','random_state','max_samples','test_size'], default=None)
    if 'criterion' in Parameter_Select:
        Criterion = st.selectbox('criterion',['gini','entropy','log_loss'])
    if 'max_depth' in Parameter_Select:
        Max_depth = st.number_input("max_depth",0,1000)
    if 'min_samples_split' in Parameter_Select:
        Min_samples_split = st.number_input("min_samples_split",0,1000)
    if 'min_samples_leaf' in Parameter_Select:
        Min_samples_leaf = st.number_input("min_samples_leaf",0,1000)
    if 'min_weight_fraction_leaf' in Parameter_Select:
        Min_weight_fraction_leaf =  st.number_input("min_weight_fraction_leaf",0,1000)
    if 'max_features' in Parameter_Select:
        Max_features = st.selectbox('max_features',['sqrt','log2'])
    if 'max_leaf_nodes' in Parameter_Select:
        Max_leaf_nodes = st.number_input("max_leaf_nodes",0,1000,5)
    if 'min_impurity_decrease' in Parameter_Select:
        Min_impurity_decrease = st.number_input("min_impurity_decrease",0,1000)
    if 'bootstrap' in Parameter_Select:
        Bootstrap = st.selectbox('criterion',['True','False'])
    if 'random_state' in Parameter_Select:
        Random_state = st.number_input("random_state",0,1000)
    if 'max_samples' in Parameter_Select:
        Max_samples = st.number_input("max_samples",0,1000)
    if 'max_samples' in Parameter_Select:
        Max_samples = st.number_input("max_samples",0,1000)
    if 'test_size' in Parameter_Select:
        num = st.number_input("test_size",min_value=0.0,max_value=1.0,step=1e-2,format="%.3f")

# Set Regression Parameters

if ML_Task == 'Regression':
    Criterion = 'squared_error'
    Max_depth = None
    Min_samples_split = 2
    Min_samples_leaf = 1
    Min_weight_fraction_leaf = 0.0
    Max_features = 1.0
    Max_leaf_nodes = None
    Min_impurity_decrease = 0.0
    Bootstrap = True
    Random_state = 42
    Max_samples= None 
    Test_size = 0.2
    Parameter_Select = st.multiselect('Select Paramters To Change', ['criterion','max_depth','min_samples_split','min_samples_leaf','min_weight_fraction_leaf','max_features','max_leaf_nodes','min_impurity_decrease','bootstrap','random_state','max_samples','test_size'], default=None)
    if 'criterion' in Parameter_Select:
        Criterion = st.selectbox('criterion',['squared_error', 'absolute_error', 'friedman_mse', 'poisson'])
    if 'max_depth' in Parameter_Select:
        Max_depth = st.number_input("max_depth",0,1000)
    if 'min_samples_split' in Parameter_Select:
        Min_samples_split = st.number_input("min_samples_split",0,1000)
    if 'min_samples_leaf' in Parameter_Select:
        Min_samples_leaf = st.number_input("min_samples_leaf",0,1000)
    if 'min_weight_fraction_leaf' in Parameter_Select:
        Min_weight_fraction_leaf =  st.number_input("min_weight_fraction_leaf",0,1000)
    if 'max_features' in Parameter_Select:
        Max_features = st.selectbox('max_features',['sqrt','log2'])
    if 'max_leaf_nodes' in Parameter_Select:
        Max_leaf_nodes = st.number_input("max_leaf_nodes",0,1000,5)
    if 'min_impurity_decrease' in Parameter_Select:
        Min_impurity_decrease = st.number_input("min_impurity_decrease",0,1000)
    if 'bootstrap' in Parameter_Select:
        Bootstrap = st.selectbox('criterion',['True','False'])
    if 'random_state' in Parameter_Select:
        Random_state = st.number_input("random_state",0,1000)
    if 'max_samples' in Parameter_Select:
        Max_samples = st.number_input("max_samples",0,1000)
    if 'test_size' in Parameter_Select:
        num = st.number_input("test_size",min_value=0.0,max_value=1.0,step=1e-2,format="%.3f")

Num_Features = st.number_input('How Many Features In The Algorithim? (Skip If Using Recursive_Feature_Elimination)',1,1000,1)

# Select Regression Feature Selection 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Test_size, random_state=Random_state)
if ML_Task == 'Regression':
    Feature_Selection_Method = st.selectbox('Which Feature Selection Method ?',['Univariate_Feature_Selection','Recursive_Feature_Elimination'])
    if Feature_Selection_Method == 'Univariate_Feature_Selection':
        Univariate_Method = st.selectbox('Which Score Function ?',['f_regression'])
        if Univariate_Method == 'f_regression':
            select_feature = SelectKBest(f_regression, k=Num_Features).fit(X_train, y_train)
            X_train_selected = select_feature.transform(X_train)
            X_test_selected = select_feature.transform(X_test)
            cols_idxs = select_feature.get_support(indices=True)
            features_df_new = X_train.iloc[:,cols_idxs]
            st.write('Feature_Used')
            features_df_new
    if Feature_Selection_Method == 'Recursive_Feature_Elimination':
        rfecv = RFECV(estimator=RandomForestRegressor(), step=1, cv=5, scoring='r2')
        rfecv = rfecv.fit(X_train, y_train)
        X_train_selected = rfecv.transform(X_train)
        X_test_selected = rfecv.transform(X_test)
        cols_idxs = rfecv.get_support(indices=True)
        features_df_new = X_train.iloc[:,cols_idxs]
        st.write('Feature_Used')
        features_df_new
if Theme_Selection == 'Cars':
    image = Image.open('RFLMUR.jpeg')
if Theme_Selection == 'Bambi':
    image = Image.open('RFBAUR.jpeg')
if Theme_Selection == 'Peter Pan':
    image = Image.open('RFPPUR.jpeg')
st.image(image)

# Select Classification Feature Selection

if ML_Task == 'Classification':
    Feature_Selection_Method = st.selectbox('Which Feature Selection Method ?',['Univariate_Feature_Selection','Recursive_Feature_Elimination'])
    if Feature_Selection_Method == 'Univariate_Feature_Selection':
        Univariate_Method = st.selectbox('Which Score Function ?',['f_classif','chi2'])
        if Univariate_Method == 'f_classif':
            select_feature = SelectKBest(f_classif, k=Num_Features).fit(X_train, y_train)
            X_train_selected = select_feature.transform(X_train)
            X_test_selected = select_feature.transform(X_test)
            cols_idxs = select_feature.get_support(indices=True)
            features_df_new = X_train.iloc[:,cols_idxs]
            st.write('Feature_Used')
            features_df_new
        if Univariate_Method == 'chi2':
            select_feature = SelectKBest(chi2, k=Num_Features).fit(X_train, y_train)
            X_train_selected = select_feature.transform(X_train)
            X_test_selected = select_feature.transform(X_test)
            cols_idxs = select_feature.get_support(indices=True)
            features_df_new = X_train.iloc[:,cols_idxs]
            st.write('Feature_Used')
            features_df_new
    if Feature_Selection_Method == 'Recursive_Feature_Elimination':
        rfecv = RFECV(estimator=RandomForestClassifier(), step=1, cv=5, scoring='accuracy')
        rfecv = rfecv.fit(X_train, y_train)
        X_train_selected = rfecv.transform(X_train)
        X_test_selected = rfecv.transform(X_test)
        cols_idxs = rfecv.get_support(indices=True)
        features_df_new = X_train.iloc[:,cols_idxs]
        st.write('Feature_Used')
        features_df_new

train_accuracies = []
test_accuracies = []

# Create and Run Regression Model

if ML_Task == 'Regression':
    n_estimators_range = range(1, 301, 10) 
    oob_errors = []
    for n_estimators in n_estimators_range:
        rf_model = RandomForestRegressor(n_estimators=n_estimators,criterion = Criterion,max_depth=None,min_samples_split=Min_samples_split,min_samples_leaf=Min_samples_leaf,min_weight_fraction_leaf=Min_weight_fraction_leaf,max_features=Max_features,max_leaf_nodes=Max_leaf_nodes,min_impurity_decrease=Min_impurity_decrease,bootstrap=Bootstrap,random_state=Random_state,max_samples=Max_samples,oob_score=True)
        rf_model.fit(X_train_selected, y_train)
        y_pred_train = rf_model.predict(X_train_selected)
        train_acc = r2_score(y_train, y_pred_train)
        train_accuracies.append(train_acc)
        y_pred = rf_model.predict(X_test_selected)
        test_acc = r2_score(y_test, y_pred)
        test_accuracies.append(test_acc)
    best_num_estimator = n_estimators_range[test_accuracies.index(max(test_accuracies))]
    num_estimators = st.number_input(f"Number of Estimators (Recommended :{best_num_estimator})",1,1000,best_num_estimator)
    rf_model = RandomForestRegressor(n_estimators=best_num_estimator,criterion = Criterion,max_depth=None,min_samples_split=Min_samples_split,min_samples_leaf=Min_samples_leaf,min_weight_fraction_leaf=Min_weight_fraction_leaf,max_features=Max_features,max_leaf_nodes=Max_leaf_nodes,min_impurity_decrease=Min_impurity_decrease,bootstrap=Bootstrap,random_state=Random_state,max_samples=Max_samples)
    rf_model.fit(X_train_selected, y_train)
    y_pred = rf_model.predict(X_test_selected)

    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

if Theme_Selection == 'Cars':
    image = Image.open('RFLMNO.jpeg')
if Theme_Selection == 'Bambi':
    image = Image.open('RFBANO.jpeg')
if Theme_Selection == 'Peter Pan':
    image = Image.open('RFPPNO.jpeg')
st.image(image)

# Create and Run Classification Model

if ML_Task == 'Classification':
    n_estimators_range = range(1, 301, 10) 
    oob_errors = []
    for n_estimators in n_estimators_range:
        rf_model = RandomForestClassifier(n_estimators=n_estimators,criterion = Criterion,max_depth=None,min_samples_split=Min_samples_split,min_samples_leaf=Min_samples_leaf,min_weight_fraction_leaf=Min_weight_fraction_leaf,max_features=Max_features,max_leaf_nodes=Max_leaf_nodes,min_impurity_decrease=Min_impurity_decrease,bootstrap=Bootstrap,random_state=Random_state,max_samples=Max_samples,oob_score=True)
        rf_model.fit(X_train_selected, y_train)
        y_pred_train = rf_model.predict(X_train_selected)
        train_acc = r2_score(y_train, y_pred_train)
        train_accuracies.append(train_acc)
        y_pred = rf_model.predict(X_test_selected)
        test_acc = r2_score(y_test, y_pred)
        test_accuracies.append(test_acc)
    best_num_estimator = n_estimators_range[test_accuracies.index(max(test_accuracies))]
    num_estimators = st.number_input(f"Number of Estimators (Recommended :{best_num_estimator})",1,1000,best_num_estimator)
    rf_model = RandomForestClassifier(n_estimators=best_num_estimator,criterion = Criterion,max_depth=None,min_samples_split=Min_samples_split,min_samples_leaf=Min_samples_leaf,min_weight_fraction_leaf=Min_weight_fraction_leaf,max_features=Max_features,max_leaf_nodes=Max_leaf_nodes,min_impurity_decrease=Min_impurity_decrease,bootstrap=Bootstrap,random_state=Random_state,max_samples=Max_samples)
    rf_model.fit(X_train_selected, y_train)
    y_pred = rf_model.predict(X_test_selected)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)


st.write('Evaluation Metrics')
st.write(f'Train Accuracy - : {rf_model.score(X_train_selected,y_train):.3f}')
st.write(f'Test Accuracy - : {rf_model.score(X_test_selected,y_test):.3f}')

# Plot Train Accuracies Over Time

plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_accuracies, label='Test R-squared (R2)')
plt.xlabel('Number of Estimators')
plt.ylabel('R-squared (R2) Score')
plt.title('Train Accuracies vs. Number of Estimators')
plt.legend()
st.pyplot(plt)

# Plot Test Accuracies Over Time

plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, test_accuracies, label='Train R-squared (R2)')
plt.xlabel('Number of Estimators')
plt.ylabel('R-squared (R2) Score')
plt.title('Test Accuracies vs. Number of Estimators')
plt.legend()
st.pyplot(plt)

# Plot MAE
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue')
plt.title(f'Actual vs. Predicted (MAE: {mae:.3f})')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# Plot Regression Line

reg_mae = LinearRegression()
reg_mae.fit(np.array(y_test).reshape(-1, 1), y_pred)
y_pred_reg_mae = reg_mae.predict(np.array(y_test).reshape(-1, 1))
plt.plot(y_test, y_pred_reg_mae, color='red', linewidth=2)

# Plot MSE

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, color='green')
plt.title(f'Actual vs. Predicted (MSE: {mse:.3f})')
plt.xlabel('Actual')
plt.ylabel('Predicted')

reg_mse = LinearRegression()
reg_mse.fit(np.array(y_test).reshape(-1, 1), y_pred)
y_pred_reg_mse = reg_mse.predict(np.array(y_test).reshape(-1, 1))
plt.plot(y_test, y_pred_reg_mse, color='orange', linewidth=2)
st.pyplot(plt)


# Select Way To Export

Prediction_Select = st.selectbox('Prediction Method',['Import Test Data','Custom'])
if Prediction_Select == 'Custom':
    Prediction_Variable_List = []
    for i in features_df_new:
        Max_samples = st.number_input(f'{i} (Input)',0,1000)
        Prediction_Variable_List.append(Max_samples)
    Prediction = rf_model.predict([Prediction_Variable_List])
    Prediction_Variable_List.extend(Prediction)
    Feature_Columns = []
    for i in features_df_new.columns:
        Feature_Columns.append(i)
    Feature_Columns.append(y_prediction)
    df_prediction_list = pd.DataFrame([Prediction_Variable_List], columns=Feature_Columns)
    st.write(df_prediction_list)
if Prediction_Select == 'Import Test Data':
    Upload2 = st.file_uploader('Upload CSV File Here', type=None, accept_multiple_files=False, key=12, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    df2 = pd.read_csv(Upload2)
    df2 = df2.drop(columns=dropped_columns)
    One_Hot_List2 = []
    for (columnName, columnData) in df2.iteritems():
        if df2[columnName].dtypes not in [int,float]:
            One_Hot_List2.append(columnName)
    df2 = pd.get_dummies(df2, columns=One_Hot_List2)
    Import_Drop_List = []
    for i in df2.columns:
        if i not in features_df_new.columns:
            Import_Drop_List.append(i)
    Import_Prediction_Array = []
    df2 = df2.drop(columns=Import_Drop_List)
    for i in features_df_new.columns:
        if i not in df2.columns:
            df2[i]=0
    for index, row in df2.iterrows():
        Import_Prediction_Array.append(list(row))
    Prediction_List = []
    for i in Import_Prediction_Array:
        Prediction = rf_model.predict([i])
        i.extend(Prediction)
        Prediction_List.append(i)
    Feature_Columns = []
    for i in features_df_new.columns:
        Feature_Columns.append(i)
    Feature_Columns.append(y_prediction)
    df_prediction_list = pd.DataFrame(Prediction_List, columns=Feature_Columns)
    st.write(df_prediction_list)
Export_Option = st.multiselect('How Do You Want To Export The Data ?',['Excel','CSV'])
if 'CSV' in Export_Option:
    CSV_Choice = st.text_input('Choose The Name')
    df_prediction_list.to_csv(f'{CSV_Choice}.csv', encoding='utf-8')
    st.write('Exported CSV File Successfully ')
if 'Excel' in Export_Option:
    Excel_Choice = st.text_input('Choose The Name')
    df_prediction_list.to_excel(f'{Excel_Choice}.xlsx', encoding='utf-8',index=False)
    st.write('Exported Excel Successfully ')














