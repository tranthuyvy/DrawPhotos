import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from io import BytesIO
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
from rembg import remove

def numpy_to_pil_image(numpy_image):
    return Image.fromarray(numpy_image)


def waterSketch(inp_img):
    # Đọc ảnh
    inp_img = np.array(inp_img, dtype="uint8")

    #Tạo hiểu ứng washout color
    img_hsv = cv2.cvtColor(inp_img, cv2.COLOR_BGR2HSV)
    ori_image = cv2.cvtColor(inp_img, cv2.COLOR_BGR2HSV)
    adjust_v = (img_hsv[:, :, 2].astype("uint") + 5) * 3
    adjust_v = ((adjust_v > 255) * 255 + (adjust_v <= 255) * adjust_v).astype("uint8")
    img_hsv[:, :, 2] = adjust_v
    tmp_bgr = img_hsv
    img_soft = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    tmp_hsv = img_soft
    img_soft = cv2.GaussianBlur(img_soft, (51, 51), 50)

    #Tạo hiệu ứng Outline Sketch
    img_gray = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    # img_filter = img_gray
    # invert = cv2.bitwise_not(img_gray)
    # blur = cv2.GaussianBlur(invert, (71, 71), 0)
    blur = cv2.GaussianBlur(img_gray, (71, 71), 0)
    # invertBlur = cv2.bitwise_not(blur)
    sketch = cv2.divide(img_gray, blur, scale=220.0)
    sketch = cv2.merge([sketch, sketch, sketch])
    img_water = ((sketch / 255.0) * img_soft).astype("uint8")

    st.subheader("Drawing Water")
    result_water = st.columns(1)
    with result_water[0]:
        st.image(img_water, width=500, caption="Final Result")

    #Display steps
    list_img = [inp_img, img_soft, img_gray, sketch, img_water]
    # list_img = [inp_img, ori_image, tmp_bgr, tmp_hsv, img_soft, img_gray, blur, sketch, img_water]

    steps_colomns = st.columns(4)
    y = 0
    for i in range(len(list_img)):
        if y >= 4:
            y = 0
        with steps_colomns[y]:
            st.image(list_img[i], width=150, caption=f"Step {i + 1}")
        y = y + 1
    return img_water

def pencilsketch(inp_img):
    gray_image = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)

    inverted_image = 250 - gray_image

    inverted_blurred = cv2.GaussianBlur(inverted_image, (11, 11), 0)

    final = cv2.divide(gray_image, 255 - inverted_blurred, scale=240)

    st.subheader("Drawing Pencil")
    final_result = st.columns(1)
    with final_result[0]:
        st.image(final, width=500, caption="Final Result")

    list_img = [inp_img, gray_image, inverted_image, inverted_blurred, final]
    steps_colomns = st.columns(4)
    y = 0
    for i in range(len(list_img)):
        if y >= 4:
            y = 0
        with steps_colomns[y]:
            st.image(list_img[i], width=120, caption=f"Step {i + 1}")
        y = y + 1
    return final

#edit image
def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    enhanced_image = enhancer.enhance(factor)
    return enhanced_image


def adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(factor)
    return enhanced_image


def adjust_sharpness(image, factor):
    enhancer = ImageEnhance.Sharpness(image)
    enhanced_image = enhancer.enhance(factor)
    return enhanced_image


def blur_image(image, radius):
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def remove_noise(image, noise_strength):
    # Làm giảm nhiễu ảnh
    return cv2.fastNlMeansDenoisingColored(np.array(image), None, noise_strength, 10, 7, 21)

def load_an_image(image_path):
    img = Image.open(image_path)
    return img
def cartoonizePhoto(img):
    image = np.asarray(img)
    K = 9
    scale_ratio = 0.95

    # Tính toán kích thước mới
    new_width = int(image.shape[1] * scale_ratio)
    new_height = int(image.shape[0] * scale_ratio)
    new_dimensions = (new_width, new_height)
    image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    # Lấy màu của từng pixel dựa trên nhãn
    res2 = centers[labels.flatten()]
    # Reshape ảnh đã phân loại
    res2 = res2.reshape(image.shape)

    contoured_image = np.copy(res2)
    gray = cv2.cvtColor(contoured_image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 2)
    edged = cv2.Canny(gray_blur, 100, 200, L2gradient=True)
    contours, hierarchy = cv2.findContours(edged,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(contoured_image, contours, contourIdx=-1, color=1, thickness=1)

    st.subheader("Cartoonize")
    result = st.columns(1)
    with result[0]:
        st.image(contoured_image, width=500, caption="Final Result")

    list_img = [image, res2, gray, gray_blur, edged, contoured_image]
    steps_colomns = st.columns(5)
    y = 0
    for i in range(len(list_img)):
        if y >= 4:
            y = 0
        with steps_colomns[y]:
            st.image(list_img[i], width=120, caption=f"Step {i + 1}")
        y = y + 1

    return contoured_image


def process_image(input_image):
    # Tách nền
    output_image = remove(input_image)

    return output_image

def flip_image(image, direction="horizontal"):
    if direction == "horizontal":
        return cv2.flip(image, 1)
    elif direction == "vertical":
        return cv2.flip(image, 0)

def apply_pixelate(image, pixel_size=10):
    h, w, _ = image.shape
    small = cv2.resize(image, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def image_editor(image, editor_option):
    if editor_option == 'Flip Horizontal':
        processed_image = flip_image(image, direction="horizontal")
    elif editor_option == 'Flip Vertical':
        processed_image = flip_image(image, direction="vertical")
    elif editor_option == 'None':
        pass

    return processed_image

# def oilPaint(img):
#     res = cv2.xphoto.oilPainting(img, 7, 1)
#     st.subheader("Oil Paint")
#     result = st.columns(1)
#     with result[0]:
#         st.image(res, width=500, caption="Final Result")
#     return res

def main():
    st.set_page_config(
        page_title="TTV | Drawing From Photo",
        page_icon="favicon.ico",
    )

    st.title('Drawing From Photos')
    st.write("This is an application developed for applying drawing.")
    st.subheader("Choose an option")

    option = st.selectbox('Choose Options',
                          (
                            'Drawing Pencil',
                            'Drawing Water Color',
                            'Cartoon',
                            'Edit Image',
                            'Pixelate',
                            'Remove Background',
                            'Free Drawing'))

    if option != 'Free Drawing':
        st.subheader("Upload image")
        image_file = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"])

        if image_file is not None:

            input_image = load_an_image(image_file)

            st.subheader("Original Image")
            st.image(load_an_image(image_file), width=500)

            if option == 'Drawing Water Color':
                result_image = waterSketch(input_image)
                buf = BytesIO()
                final_pil_image = numpy_to_pil_image(result_image)
                final_pil_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Image",
                    data=byte_im,
                    file_name="watercolor.png",
                    mime="image/png"
                )

            if option == 'Drawing Pencil':
                image = Image.open(image_file)
                final_sketch = pencilsketch(np.array(image))
                im_pil = numpy_to_pil_image(final_sketch)

                buf = BytesIO()
                im_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Image",
                    data=byte_im,
                    file_name="pencil.png",
                    mime="image/png"
                )

            if option == 'Cartoon':
                image = Image.open(image_file)
                final_sketch = cartoonizePhoto(image)
                im_pil = numpy_to_pil_image(final_sketch)

                buf = BytesIO()
                im_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Image",
                    data=byte_im,
                    file_name="cartoonize.png",
                    mime="image/png"
                )

            if option == 'Pixelate':
                img_array = np.array(input_image)
                pixel_size = st.slider("Select Pixel Size:", 1, 50, 10)
                processed_image = apply_pixelate(img_array, pixel_size)
                st.subheader("Pixelate Image")
                st.image(processed_image, width=500)

            if option == 'Remove Background':

                if image_file is not None:
                    input_image = Image.open(image_file)

                    # Xử lý ảnh
                    processed_image = process_image(input_image)
                    st.subheader("Processed Image")
                    st.image(processed_image, width=500)

            if option == 'Edit Image':

                if image_file is not None:

                    original_image = Image.open(image_file)
                    img_array = np.array(original_image)
                    #  st.image(original_image, caption="Original Image", use_column_width=True)

                    # Sidebar
                    st.sidebar.header("Adjustments")

                    brightness_factor = st.sidebar.slider("Brightness", 0.1, 3.0, 1.0)
                    contrast_factor = st.sidebar.slider("Contrast", 0.1, 3.0, 1.0)
                    sharpness_factor = st.sidebar.slider("Sharpness", 0.1, 10.0, 1.0)
                    blur_radius = st.sidebar.slider("Blur Radius", 0.0, 10.0, 0.0)


                    # Thêm thanh trượt cho giảm nhiễu
                    noise_strength = st.sidebar.slider("Noise Reduction", 0.0, 20.0, 0.0)

                    editor_option = st.sidebar.selectbox('Choose Flip Option',
                                                         ('None', 'Flip Horizontal', 'Flip Vertical'))
                    # Apply adjustments
                    adjusted_image = original_image.copy()
                    adjusted_image = adjust_brightness(adjusted_image, brightness_factor)
                    adjusted_image = adjust_contrast(adjusted_image, contrast_factor)
                    adjusted_image = adjust_sharpness(adjusted_image, sharpness_factor)
                    if blur_radius > 0:
                        adjusted_image = blur_image(adjusted_image, blur_radius)
                    adjusted_image = remove_noise(adjusted_image, noise_strength)
                    st.subheader("Image with Edit image")
                    # Display the result
                    img_array = np.array(adjusted_image)
                    im_pil = Image.fromarray(img_array)

                    if editor_option != 'None':
                        adjusted_image = image_editor(img_array, editor_option)
                    st.image(adjusted_image, width=500)

                    buf = BytesIO()
                    im_pil.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    st.download_button(
                        label="Download Image",
                        data=byte_im,
                        file_name="edit.png",
                        mime="image/png"
                    )

    if option == 'Free Drawing':
        st.subheader("Free Drawing")

        color = st.color_picker("Choose color", "#000000")

        drawing_mode = st.radio(
            "Choose drawing tool",
            ('Free Draw', 'Line', 'Rectangle'))

        if drawing_mode == 'Free Draw':
            canvas_result = st_canvas(
                fill_color=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3)",
                stroke_width=5,
                stroke_color=color,
                background_color="#eee",
                height=500,
                # width=700,
                drawing_mode="freedraw",
                key="canvas", )

        elif drawing_mode == 'Line':
            canvas_result = st_canvas(
                fill_color=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3)",
                stroke_width=5,
                stroke_color=color,
                background_color="#eee",
                height=500,
                drawing_mode="line",
                key="canvas", )
        elif drawing_mode == 'Rectangle':
            canvas_result = st_canvas(
                fill_color=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3)",
                stroke_width=5,
                stroke_color=color,
                background_color="#eee",
                height=500,
                drawing_mode="rect",
                key="canvas", )

if __name__ == '__main__':
    main()