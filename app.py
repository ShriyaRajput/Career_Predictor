# **1. Importing Necessary Libraries** 📚

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
import time
import streamlit as st
from db import *

pickleFile=open("weights.pkl","rb")
regressor=pickle.load(pickleFile) # our model

#---------------------------- Loading Dataset ----------------------------

df = pd.read_csv('./data/mldata.csv')
df.head()

df['workshops'] = df['workshops'].replace(['testing'],'Testing')
df.head()

print(df.columns.unique)

n = df['Suggested Job Role'].unique()
print(len(n))

print('The shape of our training set: %s professionals and %s features'%(df.shape[0],df.shape[1]))


# ---------------------------- Feature Engineering ----------------------------

# Binary Encoding for Categorical Variables

newdf = df
newdf.head(10)

cols = df[["self-learning capability?", "Extra-courses did","Taken inputs from seniors or elders", "worked in teams ever?", "Introvert"]]
for i in cols:
    print(i)
    cleanup_nums = {i: {"yes": 1, "no": 0}}
    df = df.replace(cleanup_nums)

print("\n\nList of Categorical features: \n" , df.select_dtypes(include=['object']).columns.tolist())

# Number Encoding for Categorical 

mycol = df[["reading and writing skills", "memory capability score"]]
for i in mycol:
    print(i)    
    cleanup_nums = {i: {"poor": 0, "medium": 1, "excellent": 2}}
    df = df.replace(cleanup_nums)

category_cols = df[['certifications', 'workshops', 'Interested subjects', 'interested career area ', 'Type of company want to settle in?', 
                    'Interested Type of Books']]
for i in category_cols:
    df[i] = df[i].astype('category')
    df[i + "_code"] = df[i].cat.codes

print("\n\nList of Categorical features: \n" , df.select_dtypes(include=['object']).columns.tolist())

# Dummy Variable Encoding

print(df['Management or Technical'].unique())
print(df['hard/smart worker'].unique())

df = pd.get_dummies(df, columns=["Management or Technical", "hard/smart worker"], prefix=["A", "B"])
df.head()

df.sort_values(by=['certifications'])

print("List of Numerical features: \n" , df.select_dtypes(include=np.number).columns.tolist())


category_cols = df[['certifications', 'workshops', 'Interested subjects', 'interested career area ', 'Type of company want to settle in?', 'Interested Type of Books']]
for i in category_cols:
  print(i)

Certifi = list(df['certifications'].unique())
print(Certifi)
certi_code = list(df['certifications_code'].unique())
print(certi_code)

Workshops = list(df['workshops'].unique())
print(Workshops)
Workshops_code = list(df['workshops_code'].unique())
print(Workshops_code)

Certi_l = list(df['certifications'].unique())
certi_code = list(df['certifications_code'].unique())
C = dict(zip(Certi_l,certi_code))

Workshops = list(df['workshops'].unique())
print(Workshops)
Workshops_code = list(df['workshops_code'].unique())
print(Workshops_code)
W = dict(zip(Workshops,Workshops_code))

Interested_subjects = list(df['Interested subjects'].unique())
print(Interested_subjects)
Interested_subjects_code = list(df['Interested subjects_code'].unique())
ISC = dict(zip(Interested_subjects,Interested_subjects_code))

interested_career_area = list(df['interested career area '].unique())
print(interested_career_area)
interested_career_area_code = list(df['interested career area _code'].unique())
ICA = dict(zip(interested_career_area,interested_career_area_code))

Typeofcompany = list(df['Type of company want to settle in?'].unique())
print(Typeofcompany)
Typeofcompany_code = list(df['Type of company want to settle in?_code'].unique())
TOCO = dict(zip(Typeofcompany,Typeofcompany_code))

Interested_Books = list(df['Interested Type of Books'].unique())
print(Interested_subjects)
Interested_Books_code = list(df['Interested Type of Books_code'].unique())
IB = dict(zip(Interested_Books,Interested_Books_code))

Range_dict = {"poor": 0, "medium": 1, "excellent": 2}
print(Range_dict)


A = 'yes'
B = 'No'
col = [A,B]
for i in col:
  if(i=='yes'):
    i = 1
  print(i)


f =[]
A = 'r programming'
clms = ['r programming',0]
for i in clms:
  for key in C:
    if(i==key):
      i = C[key]
      f.append(i)
print(f)

C = dict(zip(Certifi,certi_code))
  
print(C)

import numpy as np
array = np.array([1,2,3,4])
array.reshape(-1,1)

def inputlist(Name,Contact_Number,Email_address,
      Logical_quotient_rating, coding_skills_rating, hackathons, 
      public_speaking_points, self_learning_capability, 
      Extra_courses_did, Taken_inputs_from_seniors_or_elders,
      worked_in_teams_ever,Introvert, reading_and_writing_skills,
      memory_capability_score, smart_or_hard_work, Management_or_Techinical,
      Interested_subjects, Interested_Type_of_Books,certifications, workshops, 
      Type_of_company_want_to_settle_in, interested_career_area):
  #1,1,1,1,'Yes','Yes''Yes''Yes''Yes',"poor","poor","Smart worker", "Management","programming","Series","information security"."testing","BPA","testing"
  Afeed = [Logical_quotient_rating, coding_skills_rating, hackathons, public_speaking_points]

  input_list_col = [self_learning_capability,Extra_courses_did,Taken_inputs_from_seniors_or_elders,worked_in_teams_ever,Introvert,reading_and_writing_skills,memory_capability_score,smart_or_hard_work,Management_or_Techinical,Interested_subjects,Interested_Type_of_Books,certifications,workshops,Type_of_company_want_to_settle_in,interested_career_area]
  feed = []
  K=0
  j=0
  for i in input_list_col:
    if(i=='Yes'):
      j=2
      feed.append(j)
       
      print("feed 1",i)
    
    elif(i=="No"):
      j=3
      feed.append(j)
       
      print("feed 2",j)
    
    elif(i=='Management'):
      j=1
      k=0
      feed.append(j)
      feed.append(K)
       
      print("feed 10,11",i,j,k)

    elif(i=='Technical'):
      j=0
      k=1
      feed.append(j)
      feed.append(K)
       
      print("feed 12,13",i,j,k)

    elif(i=='Smart worker'):
      j=1
      k=0
      feed.append(j)
      feed.append(K)
       
      print("feed 14,15",i,j,k)

    elif(i=='Hard Worker'):
      j=0
      k=1
      feed.append(j)
      feed.append(K)
      print("feed 16,17",i,j,k)
    
    else:
      for key in Range_dict:
        if(i==key):
          j = Range_dict[key]
          feed.append(j)
         
          print("feed 3",i,j)

      for key in C:
        if(i==key):
          j = C[key]
          feed.append(j)
          
          print("feed 4",i,j)
      
      for key in W:
        if(i==key):
          j = W[key]
          feed.append(j)
          
          print("feed 5",i,j)
      
      for key in ISC:
        if(i==key):
          j = ISC[key]
          feed.append(j)
          
          print("feed 6",i,j)

      for key in ICA:
        if(i==key):
          j = ICA[key]
          feed.append(j)
          
          print("feed 7",i,j)

      for key in TOCO:
        if(i==key):
          j = TOCO[key]
          feed.append(j)
          
          print("feed 8",i,j)

      for key in IB:
        if(i==key):
          j = IB[key]
          feed.append(j)
          
          print("feed 9",i,j)

   
       
  t = Afeed+feed    
  output = regressor.predict([t])
  
  return(output)

def main():
    html1="""
    <div style="text-align:center; ">
      <h1> Career Predictor App </h1>
    </div>
      """
    st.markdown(html1,unsafe_allow_html=True) #simple html 
    
    st.set_page_config(page_title="Career Path Predictor", page_icon="👨🏻‍💻", layout="centered")

    # Track step in session state
    if "step" not in st.session_state:
        st.session_state.step = 1

    # Progress bar at the top
    steps = ["Personal Info", "Skills", "Preferences", "Prediction"]
    progress = (st.session_state.step - 1) / (len(steps) - 1)
    st.progress(progress)
    st.markdown(f"### Step {st.session_state.step}: {steps[st.session_state.step-1]}")

    # Step 1 - Personal Information
    if st.session_state.step == 1:
        st.text_input("Full Name", key="name")
        st.text_input("Contact Number", key="contact")
        st.text_input("Email Address", key="email")

        if st.button("Next ➡️"):
            if not st.session_state.name or not st.session_state.email:
                st.warning("Please fill out required fields (Name, Email).")
            else:
                st.session_state.step += 1
                st.rerun()

    # Step 2 - Skills
    elif st.session_state.step == 2:
        st.slider("Rate your Logical quotient Skills", 0, 10, 1, key="logical")
        st.slider("Rate your Coding Skills", 0, 10, 1, key="coding")
        st.slider("Enter number of Hackathons participated", 0, 10, 1, key="hackathons")
        st.slider("Rate Your Public Speaking", 0, 10, 1, key="public_speaking")

        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("⬅️ Back"):
                st.session_state.step -= 1
                st.rerun()
        with col2:
            if st.button("Next ➡️"):
                st.session_state.step += 1
                st.rerun()

    # Step 3 - Preferences
    elif st.session_state.step == 3:
        st.selectbox("Self Learning Capability", ("Yes", "No"), key="self_learning")
        st.selectbox("Extra Courses Taken", ("Yes", "No"), key="extra_courses")
        st.selectbox("Team Co-ordination Skill", ("Yes", "No"), key="teamwork")
        st.selectbox("Preferred Work Style", ("Smart Worker", "Hard Worker"), key="workstyle")
        st.selectbox("Interested Career Area",
            ("Testing", "System Developer", "Business Process Analyst", "Security", "Developer", "Cloud Computing"),
            key="career_area"
        )

        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("⬅️ Back"):
                st.session_state.step -= 1
                st.rerun()
        with col2:
            if st.button("Next ➡️"):
                st.session_state.step += 1
                st.rerun()

    # Step 4 - Prediction
    elif st.session_state.step == 4:
        st.subheader("🎯 Ready to Predict Your Career Path?")
        if st.button("Predict Now"):
            with st.spinner("Analyzing your responses..."):
                time.sleep(2)  # fake delay for UX
            st.success(f"Predicted Career Option: **Web Developer**")  # <- replace with model output
            st.balloons()

        if st.button("⬅️ Back"):
            st.session_state.step -= 1
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:gray;'>Developed with ❤️ by "
        "<a href='https://www.linkedin.com/in/shriiiyathakur' target='_blank'>Shriya Thakur</a></div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()