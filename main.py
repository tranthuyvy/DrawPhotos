# from sketchpy import canvas
#
# obj = canvas.sketch_from_svg('sugoku.svg',scale=30)
#
# obj.draw()

# USING PENCIL

# import cv2
#
# image = cv2.imread('Ultra.png')
#
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# inverted_gray_image = 255 - gray_image
# blurred_img = cv2.GaussianBlur(inverted_gray_image, (21, 21), 0)
# inverted_blur_image = 255 - blurred_img
# pencil_sketch_img = cv2.divide(gray_image, inverted_blur_image, scale=256.0)
#
# cv2.imshow('Old', image)
# cv2.imshow('New', pencil_sketch_img)
# cv2.waitKey(0)


import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas


def convertto_watercolorsketch(inp_img, sigma_s, sigma_r):
    if inp_img is not None:
        img_1 = cv2.edgePreservingFilter(inp_img, flags=2, sigma_s=sigma_s, sigma_r=sigma_r)
        img_water_color = cv2.stylization(img_1, sigma_s=sigma_s * 2, sigma_r=sigma_r * 2)
        return img_water_color
    else:
        return None

def pencilsketch(inp_img):
    img_pencil_sketch, pencil_color_sketch = cv2.pencilSketch(
        inp_img, sigma_s=50, sigma_r=0.05, shade_factor=0.065)
    return img_pencil_sketch

def apply_noise_reduction(image):
    return cv2.medianBlur(image, 9)

def load_an_image(image_path):
    img = Image.open(image_path)
    return img

def main():
    st.title('Drawing From Photos')
    st.write("This is an application developed for applying drawing.")
    st.subheader("Choose an option")

    option = st.selectbox('Choose Options',
                          ('Drawing Water Color',
                           'Drawing Pencil',
                           'Apply Noise',
                           'Free Drawing'))

    if option != 'Free Drawing':
        st.subheader("Upload image")
        image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])

        if image_file is not None:
            sigma_s = st.slider("Sigma_s (Scale Spatial)", 1, 200, 50)
            sigma_r = st.slider("Sigma_r (Scale Range)", 0.01, 1.0, 0.3)

            st.subheader("Image")
            st.image(load_an_image(image_file), width=300)

            if option == 'Drawing Water Color':
                image = Image.open(image_file)
                final_sketch = convertto_watercolorsketch(np.array(image), sigma_s, sigma_r)
                im_pil = Image.fromarray(final_sketch)
                st.subheader("Drawing Water Color")
                st.image(im_pil, width=300)

            if option == 'Drawing Pencil':
                image = Image.open(image_file)
                final_sketch = pencilsketch(np.array(image))
                im_pil = Image.fromarray(final_sketch)
                st.subheader("Drawing Pencil")
                st.image(im_pil, width=300)

            if option == 'Apply Noise':
                image = Image.open(image_file)
                img_array = np.array(image)
                img_array = apply_noise_reduction(img_array)
                im_pil = Image.fromarray(img_array)
                st.subheader("Image with Applied Noise")
                st.image(im_pil, width=300)

    if option == 'Free Drawing':
        st.subheader("Free Drawing")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.3)",
            stroke_width=10,
            stroke_color="rgb(0, 0, 0)",
            background_color="#eee",
            height=300,
            drawing_mode="freedraw",
            key="canvas",)

if __name__ == '__main__':
    main()

