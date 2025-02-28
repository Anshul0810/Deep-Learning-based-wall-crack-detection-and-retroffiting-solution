# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 01:42:18 2025

@author: lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 11:39:44 2025

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

import numpy as np
from keras.preprocessing import image

# Load and preprocess the test image
test_image = image.load_img('C:/Users/lenovo/Downloads/Wall Crack Detection Project/single-prediction/123456.jpg', target_size=(128, 128))  # Match CNN input size
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
    print(f"Predicted Crack Type: {prediction['type']}")
    print(f"Likely Cause of Crack: {prediction['cause']}")
    print("Retrofitting Techniques:")
    for technique in prediction['retrofit']:
        print(f"- {technique}")
else:
    print("Prediction confidence too low or unknown class!")