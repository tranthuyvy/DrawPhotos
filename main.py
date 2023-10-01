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

def convertto_watercolorsketch(inp_img):
    if inp_img is not None:
        img_1 = cv2.edgePreservingFilter(inp_img, flags = 2, sigma_s = 100, sigma_r = 0.1)
        img_water_color = cv2.stylization(img_1, sigma_s = 150, sigma_r = 0.2)
        return img_water_color
    else:
        return None

def pencilsketch(inp_img):
    img_pencil_sketch, pencil_color_sketch = cv2.pencilSketch(
        inp_img, sigma_s = 50, sigma_r = 0.05, shade_factor = 0.065)
    return (img_pencil_sketch)

def apply_noise_reduction(image):
    return cv2.medianBlur(image, 9)

def load_an_image(image):
    img = Image.open(image)
    return img

def main():
    st.title('WEBSITE DRAWING PHOTOS')
    st.write("This is an application developed for converting\
    your ***image*** to a ***Water Color Sketch*** OR ***Pencil Sketch***")
    st.subheader("Please Upload your image")

    image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])

    if image_file is not None:

        option = st.selectbox('How would you like to convert the image',
                              ('Convert to water color photos',
                               'Convert to pencil photos',
                               'Apply noise'))
        if option == 'Convert to water color photos':
            image = Image.open(image_file)
            final_sketch = convertto_watercolorsketch(np.array(image))
            im_pil = Image.fromarray(final_sketch)

            col1, col2 = st.columns(2)
            with col1:
                st.header("Image Input")
                st.image(load_an_image(image_file), width=300)

            with col2:
                st.header("Water Color Image")
                st.image(im_pil, width=300)
                buf = BytesIO()
                img = im_pil
                img.save(buf, format="JPEG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download image",
                    data=byte_im,
                    file_name="watercolorsketch.png",
                    mime="image/png"
                )

        if option == 'Convert to pencil photos':
            image = Image.open(image_file)
            final_sketch = pencilsketch(np.array(image))
            im_pil = Image.fromarray(final_sketch)

            col1, col2 = st.columns(2)
            with col1:
                st.header("Image Input")
                st.image(load_an_image(image_file), width=300)

            with col2:
                st.header("Pencil Image")
                st.image(im_pil, width=300)
                buf = BytesIO()
                img = im_pil
                img.save(buf, format="JPEG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download image",
                    data=byte_im,
                    file_name="watercolorsketch.png",
                    mime="image/png"
                )
        if option == 'Apply noise':
            image = Image.open(image_file)
            img_array = np.array(image)
            img_array = apply_noise_reduction(img_array)
            im_pil = Image.fromarray(img_array)

            col1, col2 = st.columns(2)
            with col1:
                st.header("Image Input")
                st.image(load_an_image(image_file), width=300)

            with col2:
                st.header("Image Apply Noise")
                st.image(im_pil, width=300)
                buf = BytesIO()
                img = im_pil
                img.save(buf, format="JPEG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download image",
                    data=byte_im,
                    file_name="noise.png",
                    mime="image/png"
                )


if __name__ == '__main__':
    main()
