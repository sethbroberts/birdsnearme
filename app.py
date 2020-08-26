import streamlit as st 
import numpy as np
from PIL import Image
from classify import predict

def main():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.title("Birds Around Here")

    uploaded_file = st.file_uploader("Choose an image of a bird...")
    if uploaded_file is not None:
        image = np.asarray(Image.open(uploaded_file))
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label, prob = predict(image)
        st.write(f"{label}, prob={prob*100}")

if __name__ == '__main__':
    main()