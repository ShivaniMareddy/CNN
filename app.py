import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# ---------------- PAGE ----------------

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🚧",
    layout="wide"
)

# ---------------- CSS ----------------

with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# ---------------- MODEL ----------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "road_damage_model.keras",
        compile=False
    )

model = load_model()

with open("labelmap.json","r") as f:
    label_map = json.load(f)

class_names = list(label_map.values())

# ---------------- HERO ----------------

st.markdown("""
<div class="hero">

<div style="font-size:52px">
🚧
</div>

<div class="hero-title">
AI-Based Road Damage Detection System
</div>

<div class="hero-sub">
Smart City Infrastructure Monitoring using CNN
</div>

<div class="hero-badges">
<span class="badge">CNN</span>
<span class="badge">TensorFlow</span>
<span class="badge">Computer Vision</span>
<span class="badge">Smart Infrastructure</span>
</div>

</div>
""",unsafe_allow_html=True)


# ---------------- ABOUT ----------------

st.markdown("""

<div class='panel'>

<div class='panel-label'>
ABOUT PROJECT
</div>

<h2>
📘 About Project
</h2>

<p>
Road monitoring is essential for public safety and smart city planning.
Delayed identification of potholes and cracks can increase accidents and infrastructure costs.
</p>

<p>
Convolutional Neural Networks automatically learn image patterns and classify road conditions efficiently.
</p>

<p>
<strong>Industry Applications:</strong>
</p>

<ul>
<li>Smart Cities</li>
<li>Autonomous Vehicles</li>
<li>Municipal Infrastructure</li>
<li>Traffic Safety Systems</li>
</ul>

</div>

""",unsafe_allow_html=True)


# ---------------- UPLOAD ----------------

st.markdown("""

<div class='panel'>

<div class='panel-label'>
ROAD IMAGE INPUT
</div>

<h2>
📤 Upload Road Image
</h2>

</div>

""",unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Choose image",
    type=["jpg","jpeg","png"]
)

if uploaded:

    image = Image.open(uploaded)

    col1,col2 = st.columns([1,1])

    with col1:

        st.markdown("""

<div class='panel'>

<div class='panel-label'>
IMAGE PREVIEW
</div>

<h2>🖼 Uploaded Image</h2>

</div>

""",unsafe_allow_html=True)

        st.image(
            image,
            use_container_width=True
        )

    # -------- preprocess --------

    img=image.resize((224,224))

    img=np.array(img).astype("float32")

    if len(img.shape)==2:
        img=np.stack((img,)*3,axis=-1)

    if img.shape[-1]==4:
        img=img[:,:,:3]

    img=img/255.0

    img=np.expand_dims(
        img,
        axis=0
    )

    prediction=model.predict(
        img,
        verbose=0
    )

    prediction=np.nan_to_num(
        prediction
    )[0]

    pred=np.argmax(prediction)

    confidence=float(
        np.max(prediction)*100
    )

    # -------- severity --------

    if confidence>85:

        severity="High 🔴"

    elif confidence>60:

        severity="Medium 🟠"

    else:

        severity="Low 🟢"

    with col2:

        st.markdown("""

<div class='verdict'>

<div class='verdict-title'>
Prediction Result
</div>

<div class='verdict-detail'>
CNN Classification Output
</div>

</div>

""",unsafe_allow_html=True)

        st.markdown(
f"""
<div class='metric-row'>

<div class='metric-tile'>
<span class='mt-val'>
{class_names[pred]}
</span>
<span class='mt-lbl'>
Prediction
</span>
</div>

<div class='metric-tile'>
<span class='mt-val'>
{confidence:.1f}%
</span>
<span class='mt-lbl'>
Confidence
</span>
</div>

<div class='metric-tile'>
<span class='mt-val'>
{severity}
</span>
<span class='mt-lbl'>
Severity
</span>
</div>

</div>
""",
unsafe_allow_html=True
)

        st.markdown(
"""
<h3 style='margin-top:30px'>
Probability Distribution
</h3>
""",
unsafe_allow_html=True
)

        for i in range(min(len(class_names),len(prediction))):

            width=float(prediction[i]*100)

            st.markdown(
f"""
<div class="bar-row">

<div class="bar-name">
{class_names[i]}
</div>

<div class="bar-track">
<div class="bar-fill"
style="width:{width:.1f}%;">
</div>
</div>

<div class="bar-pct">
{width:.1f}%
</div>

</div>
""",
unsafe_allow_html=True
)

# ---------------- RECOMMENDATIONS ----------------

    st.markdown("""
<div class='panel'>

<div class='panel-label'>
RECOMMENDATIONS
</div>

</div>
""",unsafe_allow_html=True)

    if class_names[pred]=="pothole":

        st.error("""
Immediate maintenance recommended.

High-risk road condition detected.
""")

    elif class_names[pred]=="crack":

        st.warning("""
Repair advised soon.

Damage may spread further.
""")

    else:

        st.success("""
Road condition stable.

Periodic monitoring recommended.
""")


# ---------------- FOOTER ----------------

st.markdown("""

<div class='info-grid'>

<div class='info-tile'>
<div class='info-title'>
🧠 Model
</div>

<div class='info-body'>
CNN trained using TensorFlow
</div>
</div>


<div class='info-tile'>
<div class='info-title'>
📊 Classes
</div>

<div class='info-body'>
Crack • Manhole • Pothole
</div>
</div>


<div class='info-tile'>
<div class='info-title'>
🏙 Smart Cities
</div>

<div class='info-body'>
AI-powered infrastructure monitoring
</div>
</div>

</div>
""",unsafe_allow_html=True)