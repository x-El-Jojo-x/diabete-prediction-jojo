import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image


st.title('Welcome to Diabetes Prediction Application using Machine Learning Algorithms') 

categories_model = ['Random_Forest', 'KNN', 'LogisticRegression','DecisionTree']
chosen_model = st.sidebar.selectbox(
   'Which Modele?',
   categories_model
)

image = Image.open('photo_diabete.png')
st.image(image,width=600)#, caption='test diabète')

file1 = open('diabetes_prediction_knn.pkl', 'rb')
knn = pickle.load(file1)
file1.close()

file2 = open('diabetes_prediction_rf.pkl', 'rb')
rf = pickle.load(file2)
file2.close()

file3 = open('diabetes_prediction_lg.pkl', 'rb')
lg = pickle.load(file3)
file3.close()

file4 = open('diabetes_prediction_dt.pkl', 'rb')
dt = pickle.load(file4)
file4.close()


data = pd.read_csv("diabete_population.csv")

#pour la normalisation################################
moy_age=data['age'].mean()
std_age=data['age'].std()
moy_grossesses=data['grossesses'].mean()
std_grossesses=data['grossesses'].std()
moy_insuline=data['insuline'].mean()
std_insuline=data['insuline'].std()
#########

print(data)
  
#age = st.number_input("Enter your age") 
age = st.slider("Select Your Age", 15, 60) 
#st.text('Selected: {}'.format(age)) 

age= (age-moy_age)/std_age #########

grossesses = float(st.radio('Select your grossesses', 
                  (['0', '1', '2', '3', '4', '5','6','7','8','9','10','+10']),horizontal=True))
print(grossesses)

grossesses = (grossesses-moy_grossesses)/std_grossesses ##############


insuline= st.number_input("Enter your insuline(mu U / ml)") 
insuline =(insuline-moy_insuline)/std_insuline ############

#moy=data['grossesses', 'age', 'insuline'].mean()
#sig=data['grossesses', 'age', 'insuline'].std()
#tn=(data['grossesses', 'age', 'insuline']-moy)/sig

if(st.button('Predict Diabete')):
    if(chosen_model == 'Random_Forest'):
        query = np.array([grossesses, age, insuline])

        query = query.reshape(1, 3)
        print(query)
        prediction = rf.predict(query)[0]
        #st.title("Predicted value " +
        #         str(prediction) + " Random_forest ")
    
    elif(chosen_model == 'KNN'):
        query = np.array([grossesses, age, insuline])

        query = query.reshape(1, 3)
        print(query)
        prediction = knn.predict(query)[0]
        #st.title("Predicted value " +
        #         str(prediction) + " KNN ")
        
    elif(chosen_model == 'LogisticRegression'):
        query = np.array([grossesses, age, insuline])

        query = query.reshape(1, 3)
        print(query)
        prediction = lg.predict(query)[0]
        #st.title("Predicted value " +
        #        str(prediction) + " Logistic regression ")
     
    elif(chosen_model == 'DecisionTree'):
        query = np.array([grossesses, age, insuline])

        query = query.reshape(1, 3)
        print(query)
        prediction = dt.predict(query)[0]
        #st.title("Predicted value " +
        #         str(prediction) + " DecisionTree ")

    if prediction == 0:
        st.title("You should have a good health!") 
    else:
        st.title("You should have Diabete")

