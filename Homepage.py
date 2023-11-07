# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st
import pdf2image
from pdf2image import convert_from_bytes
# Local Modules
import settings
import helper
import numpy as np
from PIL import Image




# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


counter=0


# Main page heading
st.title(":blue[Document Tampering Detection 🗃️]")

# Sidebar
st.sidebar.header("Configure your Model 🛠️")

# Model Options
# model_type = st.sidebar.radio(
#     "Select Task", ['Detection', 'Segmentation'])

model_type=st.sidebar.radio("Select Task",['Detection 🕵🏻'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence 😎", 0.0, 1.0, 0.40))



# Selecting Detection Or Segmentation
if model_type == 'Detection 🕵🏻':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)




# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)


object_names = list(model.names.values())
# selected_objects = st.sidebar.multiselect('Choose objects to detect', object_names, default=['Whitener'])
container = st.container()
all=st.checkbox('Select all')
select_these=None
if all:
    selected_objects=container.multiselect("Select one or more options:",object_names,object_names)
    select_these=selected_objects
    selected_indices = [object_names.index(option) for option in select_these]
else:
    selected_objects =  container.multiselect("Select one or more options:",
        object_names)
    select_these=selected_objects
    selected_indices = [object_names.index(option) for option in select_these]
    
# st.write(select_these)
source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp','pdf'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
                
            elif source_img.type=='application/pdf':
                images=pdf2image.convert_from_bytes(source_img.read())
                for page in images:
                    source_img=page
                    st.image(page,caption="Default Image",use_column_width=True)
                uploaded_image=source_img
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence,classes=selected_indices
                                    )
               
                
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                final_image = Image.fromarray(res_plotted)
                filename = f'output{counter}.png'
                final_image.save(filename)
                counter+=1
                with open("output.png", "rb") as file:
                    
                    btn = st.download_button(
                    label="Download image",
                    data=file,
                    file_name=filename,
                    mime="image/png"
          )
                # final_image=helper.downloadIt(ff)
                # st.download_button("Download Results",final_image,file_name="detected image")
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")
  


elif source_radio==settings.VIDEO:
    helper.play_stored_video(confidence,model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)


else:
    st.error("Please select a valid source type!")
