# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/cf936fbd-f2e6-43d0-8438-cf112e92c8df)
```
df.dropna()
```
![image](https://github.com/user-attachments/assets/f65ddcb0-d573-4915-b20b-c0f4c4b128b7)
```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/user-attachments/assets/93c0f771-52bb-4396-8d54-483ccbb1505f)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/461943b7-9343-4aad-b104-5ae20bf07a8b)
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/bd875e35-91d4-4dbc-a305-91fa2eb784db)
```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/122529c1-f622-4168-a095-e40ad0a421d3)
```
df1=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df1[['Height','Weight']]=scaler.fit_transform(df1[['Height','Weight']])
df1
```
![image](https://github.com/user-attachments/assets/67e8b3c3-6f22-4d7a-8a97-717415d851d3)
```
df2=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df2.head()
```
![image](https://github.com/user-attachments/assets/ad3dfbaf-ca6c-4656-a667-0ff33a3a1cf4)
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
data
```
![image](https://github.com/user-attachments/assets/4e034fa4-9ab3-4235-aadd-717e0747e835)
```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/a63bab08-98bc-47b2-b12a-b0c073ca47e4)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/aceb09d4-afad-47ec-a0b8-bd68cbec6e0f)
```
data2 = data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/d9f4ef7f-bead-4f78-926f-e2e88b4cbac0)
```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/f32443b8-55b9-4f8d-bbd1-a99019724efa)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/6436fc3d-b200-4735-9297-7cee8f97fda3)
```
data2
```
![image](https://github.com/user-attachments/assets/46003b67-ec8a-45c5-b170-0b3de189d28e)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/30ef85b2-c41d-47ab-9b58-dbaf4c82c455)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/82e35c8b-054b-4f33-b3a8-7ab6d155c075)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/3fb9ff55-4d5e-4f7e-bf42-d41fea3ad5ae)
```
y=new_data['SalStat'].values
print(y)
```
[0 0 1 ... 0 0 0]
```
x = new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/8e5c8600-3661-4413-9932-d10db48530d5)
```
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```

![image](https://github.com/user-attachments/assets/c83dfc95-2bb3-4f65-b109-e6c95ce84abd)
```
prediction = KNN_classifier.predict(test_x)
confusionMmatrix = confusion_matrix(test_y, prediction)
print(confusionMmatrix)
```
![image](https://github.com/user-attachments/assets/10406f29-26fe-4c53-986b-76a0f8c83540)
```
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)
```
0.8392087523483258
```
print('Misclassified samples: %d' % (test_y != prediction).sum())
```
Misclassified samples: 1455
```
data.shape

(31978, 13)
```
## FEATURE SELECTION TECHNIQUES
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/4f8d8ad7-c460-4e9f-82a1-555619c736fb)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/user-attachments/assets/09383a58-f5d9-4759-bf18-a800a5d0c9c6)
```
chi2, p, _, _ = chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
```
![image](https://github.com/user-attachments/assets/9ec16d78-94ce-464d-a560-951ee348d4e4)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target' :[0,1,1,0,1]
}
df=pd.DataFrame(data)
X=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif, k=1)
X_new = selector.fit_transform (X,y)
selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/89469059-2c17-45d0-aa15-0f9c67b8f9b0)
# RESULT:
To read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is successful.
       
