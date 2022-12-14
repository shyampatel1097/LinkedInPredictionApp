#### Importing our packages that we need for our app
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


#### Add header to describe app
st.markdown("# Predict Whether or Not You Are a LinkedIn User")

#### Create Income input
income_input = st.selectbox("Income (Annual)", 
             options = ["Less than $10,000",
                        "$10,000 to $19,999",
                        "$20,000 to $29,999", 
                        "$30,000 to $39,999",
                        "$40,000 to $49,999", 
                        "$50,000 to $74,999", 
                        "$75,000 to $99,999", 
                        "$100,000 to $149,999", 
                        "$150,000 or more"
                        ])


if income_input == "Less than $10,000":
    income_input = 1
elif income_input == "$10,000 to $19,999":
    income_input = 2
elif income_input == "$20,000 to $29,999":
    income_input = 3
elif income_input == "$30,000 to $39,999":
    income_input = 4
elif income_input == "$40,000 to $49,999":
    income_input = 5
elif income_input == "$50,000 to $74,999":
    income_input = 6
elif income_input == "$75,000 to $99,999":
    income_input = 7
elif income_input == "$100,000 to $149,999":
    income_input = 8
else:
    income_input = 9




#### Create Education input
education_input = st.selectbox("Education level", 
             options = ["Less Than High School",
                        "Some High School",
                        "High School Graduate",
                        "Some College, No Degree",
                        "Two Year Associates Degree",
                        "Bachelors Degree",
                        "Some Post Graduate Schooling",
                        "Postgraduate or Professional Degree"                         
                        ])


if education_input == "Less Than High School":
    education_input = 1
elif education_input == "Some High School":
    education_input = 2
elif education_input == "High School Graduate":
    education_input = 3
elif education_input == "Some College, No Degree":
    education_input = 4
elif education_input == "Two Year Associates Degree":
    education_input = 5
elif education_input == "Bachelors Degree":
    education_input = 6
elif education_input == "Some Post Graduate Schooling":
    education_input = 7
else:
    education_input = 8





#### Create Parent input
parent_input = st.selectbox("Are You a Parent", 
             options = ["I am a Parent",
                        "I am Not a Parent"])

if parent_input == "I am a Parent":
    parent_input = 1
else:
    parent_input = 0





 
#### Create Married input
married_input = st.selectbox("Are You Married", 
             options = ["I am Married",
                        "I am Not Married"])

if married_input == "I am Married":
    married_input = 1
else:
    married_input = 0









#### Create Female input
female_input = st.selectbox("Do You Identify as Female", 
             options = ["Yes",
                        "No"])

if female_input == "Yes":
    female_input = 1
else:
    female_input = 0


#### Create Age input
age_input = st.slider(label="Enter Your Age", 
          min_value=1,
          max_value=98,
          value=50)








######################################################

#Reading in the data
s = pd.read_csv('social_media_usage.csv')

#Creating the function to clean the column for target variable
def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x

# Number 3 on Project
ss = pd.DataFrame({"sm_li" : np.where(s["web1h"] > 7, np.nan,
                                      np.where(clean_sm(s['web1h']) == 1, 1, 0)), 
                   "income" : np.where(s['income'] <= 9, s['income'], np.nan), 
                   "education" : np.where(s['educ2'] <= 8, s['educ2'], np.nan), 
                   "parent" : np.where(s['par'] == 1, 1, 
                                       np.where(s['par'] == 2, 0, np.nan)), 
                   "married" : np.where(s['marital'] == 1, 1, 
                                       np.where(s['marital'] <7, 0, np.nan)), 
                   "female" : np.where(s['gender'] == 2, 1, 
                                       np.where(s['gender'] == 1, 0, np.nan)),
                   "age" : np.where(s['age'] <= 97, s['age'], np.nan)                  
                  })

ss = ss.dropna()


# Creating the Target Vector and Feature Set

y = ss["sm_li"]

X = ss[["income", "education", "parent", "married", "female", "age"]]



# Splitting the data into train and test dataset

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       
                                                    test_size=0.2,    # use 20% of the data for testing
                                                    random_state=1024) # seed for reproducibility 




# Instantiate a logisic regression model and set class_weight o balances. Fit the model with the training data
lr = LogisticRegression() #Here, we are initializing the algorithm


lr.fit(X_train, y_train) # Here, we are fitting the algorithm to the training data



# Make predictions using the testing data
y_pred = lr.predict(X_test)



# Here is the dataframe to store the variables in 
person = pd.DataFrame({
    "income": [income_input],
    "education": [education_input],
    "parent": [parent_input],
    "married": [married_input],
    "female": [female_input],
    "age": [age_input]
})

print(person)

predicted_class = lr.predict(person)
if predicted_class[0]==0:
    user_or_not = "NOT a LinkedIn User"
else: 
    user_or_not = "a LinkedIn User"



probs = lr.predict_proba(person)
percent_prob = round((probs[0][1])*100, 2)




if st.button("Click for results"):
    st.write("The probability that you are a LinkedIn User is",round((probs[0][1])*100, 2),"%")
    st.write("I think that you are", user_or_not)





######################################################



















