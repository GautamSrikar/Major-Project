


import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier



df = pd.read_csv('Dataset.csv')
df.head()





## (a) Binary Encoding for Categorical Variables


cols = df[["Self-learning Capability?","Worked in teams ever?"]]
for i in cols:
    print(i)
    cleanup_nums = {i: {"yes": 1, "no": 0}}
    df = df.replace(cleanup_nums)

print("\n\nList of Categorical features: \n" , df.select_dtypes(include=['object']).columns.tolist())

## (b) Number Encoding for Categorical 

mycol = df[["Reading and Writing Skills","Memory Capability Score"]]
for i in mycol:
    print(i)    
    cleanup_nums = {i: {"poor": 0, "medium": 1, "excellent": 2}}
    df = df.replace(cleanup_nums)

category_cols = df[['Certifications', 'Workshops', 'Interested Subjects', 'Interested Career Area', 'Type of Company want to settle in?']]
for i in category_cols:
    df[i] = df[i].astype('category')
    df[i + "_code"] = df[i].cat.codes

print("\n\nList of Categorical features: \n" , df.select_dtypes(include=['object']).columns.tolist())

## (c) Dummy Variable Encoding

print(df['Management or Technical'].unique())
print(df['Hard/Smart worker'].unique())

df = pd.get_dummies(df, columns=["Management or Technical", "Hard/Smart worker"], prefix=["A", "B"])
df.head()

df.sort_values(by=['Certifications'])

print("List of Numerical features: \n" , df.select_dtypes(include=np.number).columns.tolist())


category_cols = df[['Certifications', 'Workshops', 'Interested Subjects', 'Interested Career Area', 
                    'Type of Company want to settle in?']]
for i in category_cols:
  print(i)

Certifi = list(df['Certifications'].unique())
print(Certifi)
certi_code = list(df['Certifications_code'].unique())
print(certi_code)

Workshops = list(df['Workshops'].unique())
print(Workshops)
Workshops_code = list(df['Workshops_code'].unique())
print(Workshops_code)

Certi_l = list(df['Certifications'].unique())
C = dict(zip(Certi_l,certi_code))

Workshops = list(df['Workshops'].unique())
print("Workshops:",Workshops)
Workshops_code = list(df['Workshops_code'].unique())
print(Workshops_code)
W = dict(zip(Workshops,Workshops_code))

Interested_subjects = list(df['Interested Subjects'].unique())
print("Interested Subjects:",Interested_subjects)
Interested_subjects_code = list(df['Interested Subjects_code'].unique())
ISC = dict(zip(Interested_subjects,Interested_subjects_code))

interested_career_area = list(df['Interested Career Area'].unique())
print("Interested Career Area:",interested_career_area)
interested_career_area_code = list(df['Interested Career Area_code'].unique())
ICA = dict(zip(interested_career_area,interested_career_area_code))

Typeofcompany = list(df['Type of Company want to settle in?'].unique())
print(Typeofcompany)
Typeofcompany_code = list(df['Type of Company want to settle in?_code'].unique())
TOCO = dict(zip(Typeofcompany,Typeofcompany_code))


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

def inputlist(Logical_quotient_rating, coding_skills_rating, 
      communication_skills_rating, self_learning_capability, 
      worked_in_teams_ever,reading_and_writing_skills,
      memory_capability_score, smart_or_hard_work, Management_or_Technical,
      Interested_subjects, certifications, workshops, 
      Type_of_company_want_to_settle_in, interested_career_area):
  #1,1,1,1,'Yes','Yes''Yes''Yes''Yes',"poor","poor","Smart worker", "Management","programming","Series","information security"."testing","BPA","testing"
  Afeed = [Logical_quotient_rating, coding_skills_rating, communication_skills_rating]

  input_list_col = [self_learning_capability,worked_in_teams_ever,reading_and_writing_skills,memory_capability_score,smart_or_hard_work,Management_or_Technical,
Interested_subjects,certifications,workshops,Type_of_company_want_to_settle_in,interested_career_area]
  feed = []
  K=0
  j=0
  for i in input_list_col:
    if(i=='Yes'):
      j=2
      feed.append(j)
       
      print("feed 1",j)
    
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

   
       
  t = Afeed+feed   
  print(t) 
  # Taking all independent variable columns
  df_train_x = df[['Logical Quotient Rating', 'Coding Skills Rating', 'Communication Skills Rating','Self-learning Capability?','Worked in teams ever?', 'Reading and Writing Skills', 'Memory Capability Score', 'B_hard worker', 'B_smart worker', 'A_Management', 'A_Technical', 'Interested Subjects_code', 'Certifications_code', 'Workshops_code', 'Type of Company want to settle in?_code', 'Interested Career Area_code']]

  # Target variable column
  df_train_y = df['Suggested Job Role']

  x_train, x_test, y_train, y_test = train_test_split(df_train_x, df_train_y, test_size=0.20, random_state=42)
  
  dtree = DecisionTreeClassifier(random_state=1)
  dtree = dtree.fit(x_train, y_train)
  
  userdata = t
  output = dtree.predict([t])
 
  return(output)

def main():

  with open("styles.css") as f:
     st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)
   
  html1="""
    <div style="text-align:center; text-shadow: 3px 1px 2px purple;">
      <h1>👨🏻‍💻 Career Path Prediction app 👨🏻‍💻</h1>
    </div>
      """
  st.markdown(html1,unsafe_allow_html=True) #simple html 
  
  # Images

  col1, col2, col3 = st.columns(3)

  with col1:
      st.image("./assets/Career _Isometric.png")

  with col2:
      st.image("./assets/career.png")

  with col3:
      st.image("./assets/Career _Outline.png")

  html2="""
    <div style="text-align:center; text-shadow: 3px 1px 2px purple;">
      <h2>Your Friendly Career Advisor<h2>
    </div>
      """
  st.markdown(html2,unsafe_allow_html=True) #simple html 
 

  Logical_quotient_rating = st.slider(
    'Rate your Logical quotient Skills', 0,10,1)
  

  coding_skills_rating = st.slider(
    'Rate your Coding Skills', 0,10,1)
 
  communication_skills_rating = st.slider(
    'Rate Your Communication Skills', 0,10,1)
  

  self_learning_capability = st.selectbox(
    'Self Learning Capability',
    ('Yes', 'No')
    )

  worked_in_teams_ever = st.selectbox(
    'Worked in teams ever?',
    ('Yes', 'No')
    )
    
  reading_and_writing_skills = st.selectbox(
    'Reading and writing skills',
    ('poor','medium','excellent')
    )
  

  memory_capability_score = st.selectbox(
    'Memory capability score',
    ('poor','medium','excellent')
    )
  
  smart_or_hard_work = st.selectbox(
    'Smart or Hard Work',
    ('Hard Worker', 'Smart worker')
    )
  
  Management_or_Technical = st.selectbox(
    'Management or Technical',
    ('Management', 'Technical')
    )
  
  Interested_subjects = st.selectbox(
    'Interested Subjects',
    ('Computer Architecture', 'data engineering', 'distributed computing', 'hacking',  'IOT', 'Management', 'networks', 'parallel computing', 'programming', 'Software Engineering')
    )
  
  certifications = st.selectbox(
    'Certifications',
    ('app development', 'distro making', 'full stack', 'hadoop', 'information security', 'machine learning', 'python', 'r programming', 'shell programming')
    )
  

  workshops = st.selectbox(
    'Workshops Attended',
    ('AWS', 'data science', 'database security', 'ethical hacking', 'game development', 'system designing',  
     'Testing', 'web technologies')
    )
  
  
  Type_of_company_want_to_settle_in = st.selectbox(
    'Type of Company You Want to Settle In',
    ('BPA', 'Cloud Services', 'Finance', 'Product based', 'product development', 'SAaS services', 'Sales and Marketing', 'Service Based' 'Testing and Maintainance Services', 'Web Services')
    )
  
  interested_career_area = st.selectbox(
    'Interested Career Area',
    ('Business process analyst', 'cloud computing', 'developer', 'security', 'system developer', 'testing')
    )
  
  result=""
  
  if st.button("Predict"):
    result=inputlist(Logical_quotient_rating, coding_skills_rating,
                     communication_skills_rating, self_learning_capability, worked_in_teams_ever,
                     reading_and_writing_skills,memory_capability_score, smart_or_hard_work, 
                     Management_or_Technical,Interested_subjects,
                     certifications, workshops, Type_of_company_want_to_settle_in, interested_career_area) 


    # Balloons
    st.balloons()

    #result will be displayed if button is pressed
    st.success("Predicted Career Option : "
               "{}".format(result))


if __name__=='__main__':
    main()
