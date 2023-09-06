# start git
#pipreqs --encoding utf-8 "./"
import streamlit as st
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# Load your trained Keras model
model = tf.keras.models.load_model('emo_150epoch_27_.h5')

st.title("Emotion Detection From Image")
 
st.markdown("###### This is a very simple app to detect human emotion from the image. A simple deep learning convolutional network was trained over 10100 images of people with sad and happy faces. This model was used in the background of this app. This model can only predict two emtion of human either happy or sad, so its a binary classifier. ")
# Display upload file widget
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image using OpenCV
    model = tf.keras.models.load_model('emo_150epoch_27_.h5')
    train_image=[]
    img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    print(img)
    image_size=128
    img=cv2.resize(img,(image_size,image_size))
    train_image.append(img)
    train_image=np.array(train_image)
    x=train_image
    #visualize predicted output
    emo=model.predict(x)
    print('prediction')
    print(emo)
    if emo > 0.5: 
        st.write(f"Predicted Emotion:")
        st.markdown('# Sad') # sad close to 1 or greater than 0.5
    else:
        st.write(f"Predicted Emotion:")
        st.markdown('# Happy')

    st.image(img, caption='Original Image', use_column_width=True)
