# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 01:58:14 2025

@author: lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 11:45:34 2025

@author: lenovo
"""

import numpy as np
import pickle
import streamlit as st
from tensorflow.keras.models import load_model  # Correct model loading method
from tensorflow.keras.preprocessing import image
from PIL import Image

# loading the saved model
loaded_model = pickle.load(open('C:/Users/lenovo/Downloads/Wall Crack Detection Project/training_set.sav', 'rb')) 

def wall_crack_prediction(test_image):
    # Load and preprocess the test image
    # Match CNN input size
    
    
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255.0  # Normalize
    test_image = np.expand_dims(test_image, axis=0)

    # Predict the crack type
    result = loaded_model.predict(test_image)
    print("Prediction probabilities:", result)
    
    # Mapping of classes to crack types, causes, and retrofitting techniques
    class_mapping = {
        0: {
            'type': 'Spiderweb Cracks',
            'cause': 'Poor curing, material defects',
            'retrofit': [
                'Re-plaster the affected area with quality materials.',
                'Use non-shrink grout or fillers to seal gaps.',
                'Conduct proper curing of concrete to prevent recurrence.'
                ]
            },
        1: {
            'type': 'Diagonal Cracks',
            'cause': 'Seismic forces, differential settlement',
            'retrofit': [
                'Apply carbon fiber strips for reinforcement.',
                'Use steel mesh with mortar for re-strengthening.',
                'Re-level the structure if caused by settlement.'
                ]
            },
        2: {
            'type': 'Horizontal Cracks',
            'cause': 'Settlement, soil deformation, thermal stress',
            'retrofit': [
                'Install tie rods or straps to reinforce walls.',
                'Apply epoxy injection to bond cracks.',
                'Add support beams or bracing to reduce stress.'
                ]
            },
        3: {
            'type': 'Non-crack',
            'cause': 'Crack not present',
            'retrofit': ['Retrofitting not required.']
            },
        4: {
            'type': 'Vertical Cracks',
            'cause': 'Shrinkage, uneven foundation, overload',
            'retrofit': [
                'Use polyurethane or epoxy injection for sealing.',
                'Strengthen foundations with underpinning.',
                'Improve soil compaction around the foundation.'
                ]
            },
        5: {
            'type': 'Hairline Cracks',
            'cause': 'Shrinkage, minor settlement',
            'retrofit': [
                'Use surface sealants like acrylic or epoxy.',
                'Monitor cracks for progression over time.',
                'Improve environmental conditions to reduce shrinkage.'
                ]
            },
        6: {
            'type': 'Step Cracks',
            'cause': 'Foundation movement, weak mortar joints',
            'retrofit': [
                'Replace damaged mortar with fresh mortar.',
                'Strengthen masonry with steel reinforcement.',
                'Stabilize the foundation to prevent further movement.'
                ]
            },
        7: {
            'type': 'Wide/Deep Cracks',
            'cause': 'Structural instability, heavy loads',
            'retrofit': [
                'Install steel or concrete anchors to stabilize the structure.',
                'Use epoxy injection for sealing.',
                'Reinforce with additional steel rods or plates.'
                ]
            }
        }
    
    # Get the predicted class index
    predicted_class = np.argmax(result)
    
    # Handle predictions and display retrofitting techniques
    
    if predicted_class in class_mapping:
        prediction = class_mapping[predicted_class]
        return {
            "**Predicted Crack Type**": f"- {prediction['type']}",
            "**Likely Cause of Crack**": f"- {prediction['cause']}",
            "**Retrofitting Techniques**": "\n".join([f"- {tech}" for tech in prediction['retrofit']])
            }

    else:
        return {"Error": "Prediction confidence too low or unknown class!"}
def main():
    # Inject custom CSS for the background image and text color
    st.markdown(
        f"""
        <style>
        /* Add a full-page background image */
        .stApp {{
            background-image: url("https://images.pexels.com/photos/242236/pexels-photo-242236.jpeg");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            color: white;
        }}
        /* Ensure all text is white */
        .stMarkdown, .stTextInput, .stButton, .stFileUploader, .stImage, .stSuccess {{
            color: black !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title of the app
    st.title('Deep Learning-Based Wall Crack Detection and Retrofitting Solutions')

    # Instruction
    st.write("Upload an image of a wall to detect cracks and their likely causes.")

    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    # Display uploaded image
    if uploaded_file is not None:
        image_data = Image.open(uploaded_file)
        st.image(image_data, caption='Uploaded Image', use_column_width=True)

        # Convert the uploaded image to the format required by the model
        resized_image = image_data.resize((128, 128))  # Resize to model input size
        prediction = wall_crack_prediction(resized_image)

        # Display the prediction
        # Display the prediction results with better formatting
        if prediction:
            st.markdown("<h4 style='color:black;'>Predicted Crack Type:</h4>", unsafe_allow_html=True)
            st.markdown(f"<h5 style='color:black;'>{prediction['**Predicted Crack Type**']}</h5>", unsafe_allow_html=True)
            
            st.markdown("<h4 style='color:black;'>Likely Cause of Crack:</h4>", unsafe_allow_html=True)
            st.write(f"<h5 style='color:black;'>{prediction['**Likely Cause of Crack**']}</h5>", unsafe_allow_html=True)
            
            st.markdown("<h4 style='color:black;'>Retrofitting Techniques:</h4>", unsafe_allow_html=True)
            for technique in prediction["**Retrofitting Techniques**"].split("\n"):
                st.write(f"<h5 style='color:black;'> {technique}</h5>", unsafe_allow_html=True)
# Run the app
if __name__ == '__main__':
    main()
