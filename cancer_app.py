import streamlit as st
import pickle
import pandas as pd

# Load your model
model_svm = pickle.load(open('cancer_prediction.pkl', 'rb'))

# Define the columns/features
columns = ['texture_worst', 'perimeter_worst', 'concave points_worst',
       'radius_worst', 'concave points_mean', 'area_worst']

# Streamlit app title
st.title('Cancer Prediction App')

# Input fields for each feature
predict = []
for column in columns:
    feature = st.number_input(f"Enter {column}:", format="%.2f")
    predict.append(feature)

# Convert the list to a DataFrame
predict_df = pd.DataFrame([predict], columns=columns)

# Button to make prediction
if st.button('Predict'):
    # Make the prediction
    type_of_cancer = model_svm.predict(predict_df)
    
    # Show the result
    if type_of_cancer[0] == 1:
        st.write("Malignant - Benign")
    else:
        st.write("Benign - Malignant")
