# GENDER-CLASSIFICATION-REPORT1

# GENDER CLASSIFICATION REPORT

# REPORT OVERVIEW

The objective of this gender clacssification report is to design and evaluate a reliable model for gender classification utilizing a broad and diverse dataset.
The report provides an in-depth analysis of the model's performance on the test data,with a focus on achieving high accuracy while reducing bias to ensure fair and ethical outcomes.Additiontionally the report will offer insights into the model's strengths,identify areas needing improvement and provide recommendations for its application in real -world scenarios and initiatives.

Target Variable:Gender(Classification of Both Male and Female)


# LIBRARY USED

- Pandas as pd  
- Numpy as np  
- Sklearn(Sckit learn)



#  STEP ONE-   
 -importing of libraries

 -import pandas as pd

-import numpy as np
 

# STEP TWO-

-importing our Data
-Gender=pd.read_csv("C:/Users/kola pc/Desktop/Prediction/gender_classification_v7.csv")
Gender                        




# STEP THREE-

## REPLACING OUR GENDER COLUMN WITH 0 & 1

Gender.replace("Male",1,inplace=True)


Gender.replace("Female",0,inplace=True)

Gender 

 
# STEP FOUR-

## Checking our data type
Gender.info()

  

 # STEP FIVE-

## Splitting our Data into X and Y

X=Gender.drop(columns="gender")

Y=Gender["gender"]

 # STEP SIX-

 ## Splitting our Data For training

from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)


 
#  STEP SEVEN-

   # Testing our model with SVR 

from sklearn.svm import SVC 

sv_model=SVC()
 

#  STEP EIGHT-

   # Fitting our data into the model

sv_model.fit(x_train,y_train)
 

#  STEP NINE-

   Checking the model accuracy

sv_model.score(x_train,y_train)

#  STEP TEN-

   # Testing with RandomForest

from sklearn.ensemble import RandomForestClassifier

Rf_model=RandomForestClassifier()

#  STEP ELEVEN-
# predicting the test data

Rf_model.fit(x_train,y_train)

Rf_model.score(x_test,y_test) 

## STEP TWELVE-

#Predicting our data using random forest classifier

Rf_model.predict(x_test)


## STEP THIRTEEN-

#Putting our Y test in a dataframe

np.array([y_test])


## STEP FOURTEEN-

Assigning the test result in a variable

Machine=Rf_model.predict(x_test)



## STEP FIFTEEN-

 Creating a dataFrame for the X test
Al=x_test 

## STEP SIXTEEN-

Creating a column for our result

Al["Our Result"]=y_test

## STEP SEVENTEEN-

Creating a column for machine result
Al["Machine"]=Machine

## STEP EIGHTEEN-

Calling the Data
Al

## STEP NINETEEN-

Accuracy Report

Importing the libraries

from sklearn.metrics import accuracy_score

## STEP TWENTY-

Getting the prediction and storing it in a variable 

y_preds=Rf_model.predict(x_test)

## STEP TWENTY-0NE

Accuracy Score

print(accuracy_score(y_test,y_preds))

## STEP TWENTY-TWO

Classification Report

Importing Libraries

from sklearn.metrics import classification_report

## STEP TWENTY-THREE

Getting the classification result

print(classification_report(y_test,y_preds))

## STEP TWENTY-FOUR

Confusion matrix

Importing Libraries

from sklearn.metrics import confusion_matrix

# STEP TWENTY-FIVE

print(confusion_matrix(y_test,y_preds))

# STEP TWENTY-SIX

Importing library for confusion matrix display

from sklearn.metrics import ConfusionMatrixDisplay


#  STEP TWENTY-SEVEN

Using the estimator result for Visualization

ConfusionMatrixDisplay.from_estimator(Rf_model,X,Y)

![Screenshot (2)](https://github.com/user-attachments/assets/61785655-c77b-4aa0-adfd-27c4ceebf09e)

#  STEP TWENTY-EIGHT

Getting Results from our Predictions for Visualization

ConfusionMatrixDisplay.from_predictions(y_test,y_preds)


![Screenshot (3)](https://github.com/user-attachments/assets/bb904f89-707f-4789-b8e5-b5de39acd4a2)
