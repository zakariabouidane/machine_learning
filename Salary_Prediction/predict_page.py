import streamlit as st
import pickle
import numpy as np


with open('saved_steps.pkl', 'rb') as file:
	data = pickle.load(file)

#data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
	st.title("Software Developer Salary Prediction")

	st.write("""### we need some inforamtion to predict the salary""")

	countries = ('United Kingdom of Great Britain and Northern Ireland',
       'Netherlands', 
       'United States of America', 
       'Italy', 
       'Canada',
       'Germany', 
       'Poland', 
       'France', 
       'Brazil', 
       'Sweden', 
       'Spain',
       'India', 
       'Switzerland', 
       'Australia', 
       'Russian Federation')

	education = ('Master’s degree', 
		'Bachelor’s degree', 
		'Less than a Bachelors',
		'Post grad')


	country = st.selectbox("Country", countries)
	education = st.selectbox("Education Level", education)

	experience = st.slider("Years of Experience", 0, 50, 3)

	ok = st.button("Calculate Salary")
	if ok:
		X = np.array([[country, education, experience]])
		X[:, 0] = le_country.transform(X[:,0])
		X[:, 1] = le_education.transform(X[:,1])
		X = X.astype(float)

		salary = regressor.predict(X)
		st.subheader(f"The estimated salary is ${salary[0]:.2f}")


