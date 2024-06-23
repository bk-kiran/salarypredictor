import streamlit as st
import pickle
import numpy as np 

st.set_page_config(page_title = 'Tech Job Predictor', page_icon='👨‍💻')


def load_model(): #loading model that I developed using Jupyter notebook saved as a .pkl file
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)

    return data


data = load_model()

regressor = data["model"] 

# accessing the various fields 
le_country = data["le_country"]
le_education = data["le_education"]
le_dev = data["le_dev"]
le_remote = data["le_remote"]


st.title("Predict your Tech Job Salary!")
st.write("Fill out these fields to calculate YOUR predicted salary!")

countries = (
    'United States of America', 
    'Germany',
    'United Kingdom of Great Britain and Northern Ireland',
    'India',
    'Canada',
    'France',
    'Brazil',
    'Spain',
    'Netherlands',
    'Australia',
    'Italy',
    'Sweden',
    'Poland'
)

education_levels = (
    'Less than a Bachelor’s degree',
    'Online Course/Bootcamp without degree',
    'Bachelor’s degree',
    'Master’s degree',
    'Beyond a Master’s degree'
)

developer_roles = (
    'Developer, back-end',
    'Developer, front-end',
    'Developer, full-stack',
    'Developer, QA or test',
    'Data scientist or machine learning specialist',
    'Research & Development role',
    'System administrator',
    'Developer, desktop or enterprise applications',
    'Developer, embedded applications or devices',
    'Developer, mobile',
    'DevOps specialist',
    'Database administrator',
    'Senior Executive (C-Suite, VP, etc.)',
    'Data or business analyst',
    'Cloud infrastructure engineer',
    'Academic researcher',
    'Engineer, data', 'Engineering manager',
    'Developer, game or graphics',
    'Developer Advocate',
    'Project manager',
    'Engineer, site reliability',
    'Security professional',
    'Hardware Engineer',
    'Product manager',
    'Scientist',
    'Developer Experience',
    'Marketing or sales professional',
    'Educator', 
    'Blockchain',
    'Designer',
    'Student'
)

work_types = (
    'In-person',
    'Hybrid (some remote, some in-person)',
    'Remote'
)

# selectors for the predictor
country = st.selectbox("Country of Occupation", countries, index=None, placeholder="Select a Country")
education_level = st.selectbox("Highest Education Level", education_levels, index=None, placeholder="Select an Education Level")
developer_role = st.selectbox("Tech Role", developer_roles, index=None, placeholder="Select Tech Role")
coding_experience = st.slider("Programming Experience", 0, 40, 4)
work_experience = 0
work_type = st.selectbox("Work Type", work_types, index=None, placeholder="Select Work Type" )

result = st.button("Calculate Predicted Salary")

if result:
    X = np.array([[country, education_level, developer_role, coding_experience, work_experience, work_type]]) #creating a numpy array to store selected values
    X[:, 0] = le_country.transform(X[:,0])
    X[:, 1] = le_education.transform(X[:,1])
    X[:, 2] = le_dev.transform(X[:,2])
    X[:, 5] = le_remote.transform(X[:,5])
    X = X.astype(float)

    predicted_salary = regressor.predict(X)
    st.subheader(f"The estimated yearly salary for a {education_level} in {country} is US${predicted_salary[0]:.2f}")