import streamlit as st
import torch
import torchvision as tv
from pathlib import Path
from PIL import Image
from glob import glob
import numpy as np

st.title('Object Detection App')

st.markdown("""
We are going to build an app that allows us to choose an image,
and threshold and identify what objects are in it.

### Steps
* load images
* load object detector model
* make predictions
* display results

---
""" )

st.header('Select an image')
@st.cache(allow_output_mutation=True)
def load_data():
    filepaths = glob('app/imgs/*.jpeg')
    images = [Image.open(fp).convert("RGB") for fp in filepaths]
    names = [Path(fp).name for fp in filepaths]

    return names, images

names, images = load_data()

# Create dropdown to select an image
name = st.selectbox('', names)
idx = names.index(name)
st.image(images[idx], use_column_width=True)

st.markdown('---')
st.header('Predict on image')


@st.cache(allow_output_mutation=True)
def predict(image):
    model = tv.models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval()
    to_tensor = tv.transforms.ToTensor()
    tensor = to_tensor(image)
    pred = model([tensor])[0]
    return pred

labels = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

pred = predict(images[idx])
st.write(pred)

label_names = [labels[i] for i in pred['labels']]
scores = pred['scores'].squeeze().tolist()
masks = pred['masks']


st.markdown('---')
st.header('Define Threshold')

threshold = st.number_input('prediction threshold', min_value=0.0, max_value=1.0, value=0.5)

label_names = [ln for ln, score in zip(label_names, scores) if score > threshold]
masks = [masks[i, :, :] for i, score in enumerate(scores) if score > threshold]
st.write(label_names)

st.markdown('---')
st.header('Visualize Results')

mask_name = st.selectbox('select a label', list(set(label_names)))

def flatten_masks(masks, label_names, mask_name):
    selected_masks = [mask for ln, mask in zip(label_names, masks) if ln == mask_name]
    final_mask = torch.stack(selected_masks).max(axis=0).values
    final_mask = final_mask.detach().numpy().transpose(1, 2, 0)
    return final_mask

base_image = np.array(images[idx]) / 255.
final_mask = flatten_masks(masks, label_names, mask_name)
final_image = np.clip(base_image * 0.5 + final_mask * 0.5, a_min=0, a_max=1)

st.image(final_image, use_column_width=True)
