# -------------------- Import statements ----------------------#
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


# ----------------------- Configuration ----------------------- #

st.set_page_config(
    page_title="OcuSense",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------- Styling ----------------------- #

st.markdown("""
    <style>
    /* 1. Target the Main Title (OcuSense) */
    [data-testid="stHeaderElement"] h1, .stMarkdown h1 {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        color: #247ba0 !important;
        padding-top: 1rem !important;
    }

    /* 2. Target Subheaders (About Retinal Lesions) */
    [data-testid="stHeaderElement"] h3, .stMarkdown h3 {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #31333F !important;
        margin-top: 2rem !important;
    }

    /* 3. Global Paragraph Text (Inside main area) */
    .stApp .main .stMarkdown p {
        font-size: 1.15rem !important;
        line-height: 1.7 !important;
    }

    /* 4. Fix for Expanders (About Section) */
    .st-emotion-cache-p4m0av p {
        font-size: 1.1rem !important;
    }

    /* 5. Button Styling */
    div.stButton > button {
        background-color: #247ba0;
        color: white;
        font-size: 1.1rem !important;
        font-weight: 600;
        border-radius: 12px;
        height: 3.5em;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

CLINICAL_LABELS = {
    "he": "Haemorrhage",
    "ma": "Microaneurysm",
    "ex": "Hard Exudate",
    "se": "Soft Exudate"
}

# ----------------------- Core Logic ----------------------- #

@st.cache_resource
def get_model(path):
    return YOLO(path)

def is_valid_fundus(img):
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r_mean = np.mean(img_rgb[:, :, 0])
    b_mean = np.mean(img_rgb[:, :, 2])
    
    if b_mean == 0: b_mean = 1 
    rb_ratio = r_mean / b_mean
    
    return rb_ratio > 2.0

def process_and_detect(orig_img, pipeline_func, model_path, conf):

    # 1. Preprocess
    processed_img = pipeline_func(orig_img)
    model = get_model(model_path)
    
    # 2. Inference
    results = model.predict(processed_img, conf=conf, verbose=False)[0]
    
    # 3. Scaling logic [Detection is done on the processed image and mapped to the original image]
    h_orig, w_orig = orig_img.shape[:2]
    h_proc, w_proc = processed_img.shape[:2]
    x_scale, y_scale = w_orig / w_proc, h_orig / h_proc
    
    # 4. Annotation on Original Image [Used BGR for OpenCV consistency, converted to RGB for Streamlit]
    annotator = Annotator(orig_img.copy(), line_width=max(2, int(w_orig/500)))
    
    detection_data = []
    for box in results.boxes:
        coords = box.xyxy[0].cpu().numpy()
        # Scale coordinates back to original size
        rescaled_box = [
            coords[0] * x_scale, coords[1] * y_scale,
            coords[2] * x_scale, coords[3] * y_scale
        ]
        
        cls_id = int(box.cls[0])
        label_raw = results.names[cls_id]
        label_pretty = CLINICAL_LABELS.get(label_raw, label_raw)
        conf_val = float(box.conf[0])

        annotator.box_label(rescaled_box, f"{label_pretty} {conf_val:.2f}", 
                           color=colors(cls_id, True))
        
        detection_data.append({"Type": label_pretty, "Confidence": conf_val})

    return annotator.result(), processed_img, detection_data

# ----------------------- Sidebar UI ----------------------- #

with st.sidebar:
    st.image("assets/title_icon.jpg", use_container_width=True)
    st.title("Analysis Parameters")
    
    selected_pipeline = st.selectbox("Preprocessing Strategy", ["Baseline", "Pre-processing Pipeline A","Pre-processing Pipeline B", "Pre-processing Pipeline C"])
    
    conf_level = st.slider("Confidence Threshold", min_value=0.10, max_value=0.90, value=0.30, step=0.05, help="0.30 is the recommended best filter")
    
    st.divider()
    st.info("System Status: Active")

# ----------------------- Main App UI ----------------------- #

st.title("OcuSense: Retinal Lesion Detection")
with st.container():
    st.warning("""
**Clinical Disclaimer!** 
*  OcuSense is an **early diagnostic screening tool** designed to support early detection and provide preliminary insights into retinal health. 
*  It is **not intended to replace** the clinical judgment of a healthcare professional.
*  **Do not take medical action** or alter any treatment plan based on these results without first consulting a licensed doctor or eye specialist.
""")
st.markdown("---")

uploaded = st.file_uploader("Upload Fundus Image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded:
    # Convert upload to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    tab1, tab2 = st.tabs(["Analysis of Results", "Pipeline Inspection"])
    
    with tab1:
        # VALIDATION CHECK
        valid = is_valid_fundus(raw_img)
        
        if not valid:
            st.error("**The uploaded image is not a valid fundus photo.**")
            st.write("Please upload a valid fundus photo — a high-resolution image of the retina captured by a fundus camera.")
            st.image(raw_img, channels="BGR", width=400)
        else:
            col_img, col_data = st.columns([2, 1])
        
            if st.button("Check for Retinal lesions", type="primary"):
                from preprocessing.preprocessing_baseline import preprocess_baseline
                from preprocessing.preprocessing_A import preprocess_A
                from preprocessing.preprocessing_B import preprocess_B
                from preprocessing.preprocessing_C import preprocess_C
                
                # 1. Select the preprocessing function
                if selected_pipeline == "Baseline":
                    p_func = preprocess_baseline
                elif selected_pipeline == "Pre-processing Pipeline A":
                    p_func = preprocess_A
                elif selected_pipeline == "Pre-processing Pipeline B": 
                    p_func = preprocess_B
                else:
                    p_func = preprocess_C

                # 2. Select the corresponding model path
                if selected_pipeline == "Baseline":
                    m_path = "models/baseline_best.pt"
                elif selected_pipeline == "Pre-processing Pipeline A": 
                    m_path = "models/preprocessing_A_best.pt"
                elif selected_pipeline == "Pre-processing Pipeline B": 
                    m_path = "models/preprocessing_B_best.pt"
                else:
                    m_path = "models/preprocessing_C_best.pt"
                
                with st.spinner("Analysing..."):
                    final_plot, proc_view, detections = process_and_detect(raw_img, p_func, m_path, conf_level)
                
                with col_img:
                    st.subheader("Lesion Mapping")
                    # Convert BGR to RGB for Streamlit display
                    st.image(final_plot, channels="BGR", use_container_width=True, caption="Annotated Image")
                    
                with col_data:
                    st.subheader("Summary")
                    if detections:
                        df = pd.DataFrame(detections)
                        st.metric("Total Lesions Found", len(df))
                        
                        # Grouped counts
                        summary = df['Type'].value_counts().reset_index()
                        summary.columns = ['Lesion', 'Count']
                        st.table(summary)
                        
                        with st.expander("Detailed Confidence Scores"):
                            st.dataframe(df.sort_values(by="Confidence", ascending=False), hide_index=True)
                    else:
                        st.success("No retinal lesions detected.")
                
                st.warning("⚠️ **Important!**   This analysis is generated by an experimental AI model. It is intended for screening support, not definitive diagnosis. Always verify findings with a medical professional.")

    with tab2:
        st.subheader("Preprocessing Comparison")
        c1, c2 = st.columns(2)
        c1.image(raw_img, channels="BGR", caption="Input Image", use_container_width=True)
        # Note: You need to run inference to get 'proc_view'
        try:
            c2.image(proc_view, channels="BGR", caption=f"Pipeline: {selected_pipeline}", use_container_width=True)
        except NameError:
            st.warning("Run analysis to see processed output.")

# ----------------------- About Retinal Lesions Section ----------------------- #
st.divider()
st.subheader("About Retinal Lesions")

with st.expander("Learn more"):

    st.markdown("#### 1. Types of Retinal Lesions")
    
    # Using columns to explain lesion types clearly
    l_col1, l_col2 = st.columns(2)
    
    with l_col1:
        st.markdown("""
        **Microaneurysms (MA)**  
        * Small, round outpouchings of retinal capillaries caused by weakening of vessel walls.  
        * Typically the **earliest detectable sign** of diabetic retinal damage.

        **Haemorrhages (HE)**  
        * Occur due to rupture of damaged retinal blood vessels.  
        * Appear as red spots or blotches and often indicate **progression from microaneurysms**.  
        * Microaneurysms and haemorrhages are collectively referred to as **red lesions**.
        """)

    with l_col2:
        st.markdown("""
        **Hard Exudates (EX)**  
        * Lipid and protein deposits that leak from compromised blood vessels.  
        * Appear as small, well-defined yellow or white spots.

        **Soft Exudates / Cotton Wool Spots (SE)**  
        * Result from localised retinal ischemia affecting the nerve fiber layer.  
        * Appear as white, fluffy, cloud-like patches.  
        * Both hard and soft exudates are categorized as **white lesions**.
        """)

    st.divider()

    st.markdown("#### 2. Risk Factors and Disease Progression")
    st.info("The following factors significantly increase the risk and severity of retinal lesion development:")

    r_col1, r_col2 = st.columns(2)

    with r_col1:
        st.markdown("""
        * **Long-standing Diabetes:** Prolonged exposure to high blood glucose weakens retinal vessels.
        * **Hypertension:** Elevated blood pressure increases vascular stress and leakage risk.
        * **Smoking:** Impairs oxygen delivery and accelerates microvascular damage.
        """)

    with r_col2:
        st.markdown("""
        * **High Cholesterol Levels:** Promote the formation of hard exudates.
        * **Poor Diet and Dehydration:** Excessive salt intake worsens blood pressure control.
        * **Sedentary Lifestyle:** Contributes to systemic inflammation and vascular dysfunction.
        """)

    st.divider()

    st.markdown("#### 3. Importance of Early Diagnosis")
    st.success("""
    Diabetic Retinopathy is the **leading cause of preventable vision loss among working-age adults worldwide.**  
    
    Retinal lesions often develop **without noticeable symptoms** in the early stages, meaning significant damage may occur before vision changes are perceived.

    Early screening and lesion-level detection enable timely clinical intervention, reducing the risk of irreversible vision loss and complications such as macular edema.  

    """)


# ----------------------- Footer ----------------------- #
st.markdown("---")
st.caption(" © OcuSense | 2026")