import streamlit as st
import os
import tempfile
import pandas as pd
import numpy as np
from visualize_gradcam import visualize_gradcam
from inference_engine import InferenceEngine, generate_aggregate_viz
import config

st.set_page_config(page_title="Brain Tumor MGMT Classifier", layout="wide")

st.title("🧠 Brain Tumor MGMT Radiogenomics Classifier")
st.markdown("""
This application predicts the probability of **MGMT promoter methylation** in brain tumors using 3D MRI scans 
and visualizes the model's focus areas using **Grad-CAM heatmaps**.
""")

# Tabs for different input methods
tab1, tab2 = st.tabs(["Dataset Explorer", "Zip Inference Engine"])

with tab1:
    # Sidebar for Inputs
    st.sidebar.header("Dataset Settings")

    # Get list of patients
    input_dir = "../input/train"
    if os.path.exists(input_dir):
        patients = sorted(os.listdir(input_dir))
    else:
        patients = ["No patients found"]

    selected_patient = st.sidebar.selectbox("Select Patient BraTS21ID", patients)
    mri_type = st.sidebar.selectbox("Select MRI Type (Heatmaps target FLAIR by default)", ["FLAIR", "T1w", "T1wCE", "T2w"], index=0, key="explorer_mri")
    selected_fold = st.sidebar.slider("Select Model Fold", 0, 4, 3, key="explorer_fold")

    # Session State for Explorer
    if "explorer_result" not in st.session_state:
        st.session_state.explorer_result = None

    if st.sidebar.button("Run Analysis", key="explorer_btn"):
        with st.spinner(f"Analyzing {selected_patient} ({mri_type})..."):
            try:
                # Run visualization and prediction
                case_id = int(selected_patient)
                heatmap_path, mgmt_prob, slice_img, heatmap_data = visualize_gradcam(case_id, mri_type, selected_fold)
                
                # Store in session state
                st.session_state.explorer_result = {
                    "heatmap_path": heatmap_path,
                    "mgmt_prob": mgmt_prob,
                    "slice_img": slice_img,
                    "heatmap_data": heatmap_data,
                    "mri_type": mri_type,
                    "fold": selected_fold,
                    "patient": selected_patient
                }
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

    # Display Explorer Results if available
    if st.session_state.explorer_result:
        res = st.session_state.explorer_result
        
        # Display Results
        st.divider()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Prediction")
            st.metric("MGMT Probability", f"{res['mgmt_prob']:.2%}")
            
            if res['mgmt_prob'] > 0.5:
                st.success("High Probability of MGMT Methylation")
            else:
                st.warning("Low Probability of MGMT Methylation")
            
            st.info(f"Methodology: Single-slice inference on the largest {res['mri_type']} scan.")
            st.caption(f"Using Fold {res['fold']} weights.")
            st.caption(f"Patient: {res['patient']}")

        with col2:
            st.subheader(f"Interactive Visualization ({res['mri_type']})")
            
            # Slider for Opacity
            alpha = st.slider("Heatmap Opacity", 0.0, 1.0, 0.4, step=0.05, key="explorer_alpha")
            
            # Dynamic Overlay
            import cv2
            import numpy as np
            
            # Prepare heatmap color
            heatmap_data = res['heatmap_data']
            heatmap_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min() + 1e-8)
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_norm), cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            
            # Prepare background
            img_colored = np.stack([res['slice_img']]*3, axis=-1)
            img_colored = (img_colored - img_colored.min()) / (img_colored.max() - img_colored.min() + 1e-8)
            img_colored = np.uint8(255 * img_colored)
            
            # Blend
            overlay = cv2.addWeighted(img_colored, 1.0 - alpha, heatmap_color, alpha, 0)
            
            st.image(overlay, use_column_width=True, caption=f"Overlay on Largest {res['mri_type']} Slice (Opacity: {alpha:.2f})")

with tab2:
    st.header("Upload MRI Scan (Zip)")
    st.markdown("""
    Upload a `.zip` file containing DICOM images. The engine will:
    1. **Target FLAIR**: Automatically filters for FLAIR scans for heatmap generation.
    2. **Auto-Center**: Identifies the largest anatomical slice.
    3. **Single-Slice Accuracy**: Runs inference on the largest slice (replicated to 3D).
    4. **Precise Visualization**: Shows the Grad-CAM overlay strictly on the largest slice.
    """)
    
    uploaded_file = st.file_uploader("Choose a zip file", type="zip")
    zip_fold = st.select_slider("Select Model Fold for Inference", options=[0, 1, 2, 3, 4], value=3)
    
    # Session State for Inference
    if "inference_result" not in st.session_state:
        st.session_state.inference_result = None
    
    if uploaded_file is not None:
        if st.button("Start Processing"):
            # Clear previous result
            st.session_state.inference_result = None
            
            # Save uploaded file to temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                tmp.write(uploaded_file.getvalue())
                zip_path = tmp.name
            
            try:
                with st.spinner("Processing Zip file (Auto-selecting best MRI type) and running inference..."):
                    engine = InferenceEngine(fold=zip_fold)
                    prob, heatmap, largest_img, selected_mri_type = engine.process_zip(zip_path)
                    
                    st.session_state.inference_result = {
                        "prob": prob,
                        "heatmap": heatmap,
                        "largest_img": largest_img,
                        "selected_mri_type": selected_mri_type
                    }
                    
            except Exception as e:
                st.error(f"Inference Error: {str(e)}")
            finally:
                if os.path.exists(zip_path):
                    os.remove(zip_path)

    # Display Inference Results if available
    if st.session_state.inference_result:
        res = st.session_state.inference_result
        
        st.divider()
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.subheader("Inference Result")
            st.metric("MGMT Prob (Largest Slice)", f"{res['prob']:.2%}")
            st.progress(res['prob'])
            
            if res['prob'] > 0.5:
                st.success("Result: Likely MGMT Methylation")
            else:
                st.warning("Result: Unlikely MGMT Methylation")
                
            st.write("**Methodology**:")
            st.info(f"Automatically selected **{res['selected_mri_type']}** scan based on quality priority.")
            st.write("- Identified Largest Anatomical Slice")
            st.write("- Single-Slice 3D Emulation")

        with c2:
            st.subheader(f"Interactive Visualization ({res['selected_mri_type']})")
            
            # Slider for Opacity
            alpha_zip = st.slider("Heatmap Opacity", 0.0, 1.0, 0.4, step=0.05, key="zip_alpha")
            
            # Dynamic Overlay (Reusing logic)
            import cv2
            import numpy as np
            
            heatmap = res['heatmap']
            heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_norm), cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            
            img_colored = np.stack([res['largest_img']]*3, axis=-1)
            img_colored = (img_colored - img_colored.min()) / (img_colored.max() - img_colored.min() + 1e-8)
            img_colored = np.uint8(255 * img_colored)
            
            overlay = cv2.addWeighted(img_colored, 1.0 - alpha_zip, heatmap_color, alpha_zip, 0)
            
            st.image(overlay, use_column_width=True, caption=f"Grad-CAM Overlay on Peak {res['selected_mri_type']} Slice")

# Footer
st.divider()
st.caption("Developed for Brain Tumor Radiogenomics Classification")
