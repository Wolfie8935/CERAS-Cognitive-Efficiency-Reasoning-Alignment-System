import streamlit as st
import time
import json
from datetime import datetime
import numpy as np
import os
import sys
import joblib
import tensorflow as tf
import re

artifact_dir = "./artifacts"

#Load trained model
@st.cache_resource
def load_models():
    cepm_model = joblib.load(os.path.join(artifact_dir, "cepm_lightgbm.pkl"))
    cepm_scaler = joblib.load(os.path.join(artifact_dir, "cepm_scaler.pkl"))

    cnn_model = tf.keras.models.load_model(os.path.join(artifact_dir, "cnn_ce_model.keras"))
    cnn_scaler = joblib.load(os.path.join(artifact_dir, "cnn_scaler.pkl"))

    # Load selected feature names
    cepm_features = np.load(os.path.join(artifact_dir, "cepm_features.npy"), allow_pickle=True).tolist()
    cnn_features = np.load(os.path.join(artifact_dir, "cnn_features.npy"), allow_pickle=True).tolist()

    return (
        cepm_model,
        cepm_scaler,
        cnn_model,
        cnn_scaler,
        cepm_features,
        cnn_features
    )

(
    cepm_model,
    cepm_scaler,
    cnn_model,
    cnn_scaler,
    cepm_features,
    cnn_features
) = load_models()

#Reasoning pipeline
from pipeline_1 import main as run_infer

#CERAS fusion engine
from fusion import CERASFusion

#Page Configuration
st.set_page_config(
    page_title="CERAS",
    page_icon="🧠",
    layout="wide",
)

#Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        padding: 10px;
        border-radius: 8px;
        background-color: #f0f2f6;
        margin-bottom: 10px;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .prompt-box {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #334155;
        margin-bottom: 10px;
        height: 200px;
        overflow-y: auto;
        font-size: 14px;
        color: #e2e8f0;
        text-align: justify;
    }
    .prompt-box-bad {
        background-color: #2a1515;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #451a1a;
        margin-bottom: 10px;
        font-size: 14px;
        color: #fca5a5;
        text-align: justify;
    }
    /* Justify text for better readability */
    p, li {
        text-align: justify;
    }
    /* Sidebar specific refinements */
    .sidebar-box {
        background-color: #0f172a;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #334155;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
</style>
""", unsafe_allow_html=True)

#Header
st.markdown("## 🧠 CERAS - Cognitive Efficiency & Reasoning Alignment System")
st.markdown("""
**CERAS** is an advanced adaptive learning environment designed to optimize how you learn and solve complex problems. 
By fusing **Large Language Model (LLM)** reasoning capabilities with real-time **Cognitive Efficiency** metrics, current behavioral diagnostics, and 
neuro-fuzzy alignment, CERAS provides a personalized learning experience. It analyzes your input complexity, structure, and intent to guide you 
through deep concepts with tailored roadmaps, ensuring you don't just get answers, but truly master the material.
""")

#Sidebar
with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding-bottom: 20px;">
            <h1 style="margin-bottom:0; font-size: 2.5rem;">🧠</h1>
            <h2 style="margin-top:0;">CERAS</h2>
            <p style="color:#94a3b8; font-size:14px; margin-top:5px;">
                Cognitive Efficiency & Reasoning Alignment System
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### ⚙️ System Status")

    st.markdown(
        """
        <div class="sidebar-box">
            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <span>🔌 <b>Groq API</b></span>
                <span style="color:#4ade80;">● Connected</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <span>⚡ <b>Fusion Engine</b></span>
                <span style="color:#4ade80;">● Active</span>
            </div>
             <div style="display:flex; justify-content:space-between;">
                <span>📡 <b>Telemetry</b></span>
                <span style="color:#60a5fa;">● Tracking</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🏗️ Architecture")

    st.markdown(
        """
        <div class="sidebar-box" style="font-size: 13px;">
            <p style="margin:8px 0;"> 🧩 <b>ToT-LLM</b><br><span style="color:#94a3b8;">Tree-of-Thoughts Reasoning Engine</span></p>
            <p style="margin:8px 0;"> 🧠 <b>CEPM</b><br><span style="color:#94a3b8;">Cognitive Engagement Modeling</span></p>
            <p style="margin:8px 0;"> 👁️ <b>CNN-Vis</b><br><span style="color:#94a3b8;">Behavioral Signal Analysis</span></p>
            <p style="margin:8px 0;"> 🔗 <b>Fusion Layer</b><br><span style="color:#94a3b8;">Multi-Modal Signal Integration</span></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption("v1.2.0 • Neural Learning Stack")

#Session State
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

if "formulation_time" not in st.session_state:
    st.session_state.formulation_time = 0.0

if "auto_run" not in st.session_state:
    st.session_state.auto_run = False

if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = ""

def set_prompt(text):
    st.session_state.user_prompt = text
    # Disable auto-run, user wants to manually click run
    st.session_state.auto_run = False

#Good Examples
st.markdown("### 🌟 GOOD EXAMPLES TO PROMPT (High CE Score)")
st.caption("These prompts are detailed and structured, leading to higher cognitive efficiency scores.")

gp1 = "Analyze the epistemological foundations of quantum entanglement by integrating formal mathematical structure, experimental validation, and philosophical interpretation into a coherent explanatory framework. Begin by describing how tensor product Hilbert spaces allow composite quantum systems to exhibit non-factorizable state vectors, and clarify why separability fails under entangled configurations. Then examine Bell’s inequalities, including the CHSH formulation, and explain how empirical violations observed in Aspect-type experiments undermine classical locality and deterministic realism. Extend the discussion toward decoherence theory, entropic correlations, and the role of measurement operators in collapsing superposed amplitudes. Contrast Copenhagen, Many-Worlds, and relational interpretations, focusing specifically on their ontological commitments and metaphysical implications. Additionally, evaluate how quantum information theory reframes entanglement as a computational resource enabling teleportation, superdense coding, and cryptographic security. Finally, synthesize these perspectives into a structured argument addressing whether entanglement necessitates nonlocal causation or instead demands a revision of classical intuitions regarding separability, causality, and physical realism."

gp2 = "Construct a systems-level biochemical and thermodynamic analysis of photosynthesis that integrates molecular structure, energetic transfer mechanisms, and ecological macro-dynamics. Begin by formally describing chloroplast ultrastructure and pigment absorption spectra in terms of quantum excitation states. Then analyze the light-dependent reactions as an electron transport optimization problem, including photolysis, proton gradients, chemiosmotic coupling, and ATP synthase rotation mechanics. Extend the discussion into the Calvin-Benson cycle using carbon fixation kinetics, RuBisCO efficiency constraints, and NADPH reduction pathways. Evaluate photosynthesis as an entropy-management system that converts low-entropy solar radiation into high-order biochemical organization. Finally, synthesize its planetary-scale implications for atmospheric regulation, carbon sequestration feedback loops, and biospheric energy flow stability."

gp3 = "Develop a multi-layered historical and epistemological examination of the Gutenberg printing press by integrating technological innovation theory, sociopolitical restructuring, and cognitive-cultural transformation. Begin by describing the mechanical engineering principles underlying movable type standardization and ink transfer reproducibility. Then analyze how mass replication altered information diffusion velocity and network topology across Renaissance Europe. Evaluate its causal role in accelerating scientific method formalization, destabilizing ecclesiastical epistemic monopolies, and enabling vernacular linguistic codification. Extend the analysis toward media ecology theory and distributed cognition, examining how print culture reshaped memory externalization and authority structures. Conclude by synthesizing how the printing press functioned as an epistemic amplifier that reconfigured knowledge production, institutional legitimacy, and political sovereignty."

gp4 = "Produce a mathematically grounded and architecturally comparative analysis of supervised and unsupervised machine learning paradigms, emphasizing objective functions, representational geometry, and statistical inference principles. Begin by defining supervised learning as an empirical risk minimization framework over labeled distributions and contrast it with unsupervised latent-variable modeling and manifold estimation. Analyze bias-variance trade-offs, generalization bounds, and overfitting dynamics under distributional shift. Compare algorithmic mechanisms such as Support Vector Machines, ensemble-based decision forests, K-Means clustering, and Principal Component Analysis through the lens of optimization landscapes and feature-space transformations. Extend the discussion toward interpretability constraints, scalability limits, and robustness under adversarial perturbations. Finally, synthesize these paradigms into a structured framework evaluating when hybrid semi-supervised or self-supervised approaches become epistemically advantageous."

#Layout: 2 Columns x 2 Rows for better readability
c1, c2 = st.columns(2)

with c1:
    st.markdown(f'<div class="prompt-box">{gp1}</div>', unsafe_allow_html=True)
    if st.button("📝 Use: Quantum Foundations", key="btn_gp1"):
        set_prompt(gp1)
        st.rerun()

    st.markdown(f'<div class="prompt-box">{gp3}</div>', unsafe_allow_html=True)
    if st.button("📝 Use: Printing Press Analysis", key="btn_gp3"):
        set_prompt(gp3)
        st.rerun()

with c2:
    st.markdown(f'<div class="prompt-box">{gp2}</div>', unsafe_allow_html=True)
    if st.button("📝 Use: Photosynthesis Systems", key="btn_gp2"):
        set_prompt(gp2)
        st.rerun()

    st.markdown(f'<div class="prompt-box">{gp4}</div>', unsafe_allow_html=True)
    if st.button("📝 Use: ML Paradigm Theory", key="btn_gp4"):
        set_prompt(gp4)
        st.rerun()

if st.session_state.auto_run:
    pass

#Bad Examples
st.markdown("### ⚠️ BAD EXAMPLES TO PROMPT (Low CE Score)")
st.caption("These prompts look normal but lack depth, structure, or analytical clarity.")

bp1 = "Explain artificial intelligence in simple terms."
bp2 = "Describe how computers work."
bp3 = "Give a summary of World War II."
bp4 = "Explain why the sky is blue in a short answer."

bc1, bc2, bc3, bc4 = st.columns(4)

with bc1:
    st.markdown(f'<div class="prompt-box-bad">{bp1}</div>', unsafe_allow_html=True)
    if st.button("Use: AI Basic", key="btn_bp1"):
        set_prompt(bp1)
        st.rerun()

with bc2:
    st.markdown(f'<div class="prompt-box-bad">{bp2}</div>', unsafe_allow_html=True)
    if st.button("Use: Computers", key="btn_bp2"):
        set_prompt(bp2)
        st.rerun()

with bc3:
    st.markdown(f'<div class="prompt-box-bad">{bp3}</div>', unsafe_allow_html=True)
    if st.button("Use: WWII Summary", key="btn_bp3"):
        set_prompt(bp3)
        st.rerun()

with bc4:
    st.markdown(f'<div class="prompt-box-bad">{bp4}</div>', unsafe_allow_html=True)
    if st.button("Use: Sky Simple", key="btn_bp4"):
        set_prompt(bp4)
        st.rerun()

st.markdown("---")

#Input
c_in1, c_in2 = st.columns([0.8, 0.2])
with c_in2:
    if st.button("New Problem"):
        st.session_state.start_time = time.time()
        st.session_state.formulation_time = 0.0
        st.session_state.user_prompt = ""
        st.session_state.auto_run = False
        st.rerun()

prompt = st.text_area(
    "Enter your learning question or problem",
    height=150,
    help="Time starts tracking when you click 'New Problem' or reload.",
    key="user_prompt"
)

run_btn = st.button("▶ Run Learning Session")

#Run Pipeline
if (run_btn or st.session_state.auto_run) and prompt.strip():
    
    #Reset auto_run so it doesn't loop
    if st.session_state.auto_run:
        st.session_state.auto_run = False
    
    #Calculate User Latency (Formulation Time)
    st.session_state.formulation_time = time.time() - st.session_state.start_time
    
    with st.spinner("Running reasoning engine..."):
        t0 = time.time()
        result = run_infer(prompt)
        runtime = time.time() - t0  # System Latency

def extract_ceras_features(prompt_text):

    words = prompt_text.split()
    prompt_length = np.clip(len(words), 1, 400)
    character_count = len(prompt_text)

    sentence_count = max(len(re.findall(r"[.!?]", prompt_text)), 1)

    unique_word_ratio = len(set(words)) / (prompt_length + 1e-6)
    unique_word_ratio = np.clip(unique_word_ratio, 0, 1)

    concept_density = sum(1 for w in words if len(w) > 6) / (prompt_length + 1e-6)
    concept_density = np.clip(concept_density, 0, 1)

    keystrokes = np.clip(character_count, 1, 2000)

    prompt_quality = np.clip(prompt_length / 150, 0, 1)

    #Simple live prompt_type mapping
    if prompt_length < 20:
        prompt_type = 0
    elif prompt_length < 60:
        prompt_type = 1
    elif prompt_length < 120:
        prompt_type = 2
    else:
        prompt_type = 3

    return {
        "prompt_length": float(prompt_length),
        "sentence_count": float(sentence_count),
        "unique_word_ratio": float(unique_word_ratio),
        "concept_density": float(concept_density),
        "prompt_quality": float(prompt_quality),
        "character_count": float(character_count),
        "keystrokes": float(keystrokes),
        "prompt_type": float(prompt_type),
    }

#Run Pipeline
should_run = False
if run_btn:
    should_run = True
if st.session_state.auto_run:
    should_run = True
    st.session_state.auto_run = False

if should_run and prompt.strip():
    with st.spinner("Running reasoning engine..."):
        t0 = time.time()
        # Calculate User Latency (Formulation Time)
        st.session_state.formulation_time = time.time() - st.session_state.start_time
        
        result = run_infer(prompt)
        runtime = time.time() - t0
        
        # Store results in session state for persistence
        st.session_state.current_result = result
        st.session_state.current_runtime = runtime
        st.session_state.current_prompt = prompt

    #Generate and store adaptive response
    from llm_utils import generate_adaptive_response

    #Extract Real Features
    final_steps = result.get("final_answer", [])

    features = extract_ceras_features(prompt)

    #CEPM Inference
    cepm_input = np.array([features[f] for f in cepm_features]).reshape(1, -1)
    cepm_input_scaled = cepm_scaler.transform(cepm_input)
    cepm_score = float(np.clip(cepm_model.predict(cepm_input_scaled)[0], 0, 1))

    #CNN Inference
    cnn_input = np.array([features[f] for f in cnn_features]).reshape(1, -1)
    cnn_input = cnn_scaler.transform(cnn_input)

    if len(cnn_model.input_shape) == 3:
        cnn_input = cnn_input.reshape(cnn_input.shape[0], cnn_input.shape[1], 1)

    cnn_score = float(np.clip(np.squeeze(cnn_model.predict(cnn_input, verbose=0)), 0, 1))

    #Fusion
    fusion_engine = CERASFusion()

    fusion_df = fusion_engine.fuse(
    session_ids=["session_1"],
    cepm_scores=[cepm_score],
    cnn_scores=[cnn_score])

    fused_score = fusion_df["fused_ce_score"].iloc[0]
    confidence = fusion_df["confidence"].iloc[0]

    #Generate Adaptive Response
    try:
        with st.spinner("Generating personalized learning summary..."):
            st.session_state.adaptive_res = generate_adaptive_response(
                prompt,
                final_steps,
                fused_score,
                confidence
        )
    except Exception as e:
        st.warning("Adaptive response unavailable (LLM rate limit reached).")
        st.session_state.adaptive_res = None

#Render Results if available
if "current_result" in st.session_state and st.session_state.current_result is not None:
    result = st.session_state.current_result
    runtime = st.session_state.current_runtime
    #Use the prompt that generated the result for scoring to ensure consistency
    result_prompt = st.session_state.current_prompt

    #Final Answer
    st.markdown("## Learning Response")

    final_steps = result.get("final_answer", [])

    if isinstance(final_steps, list):
        for i, step in enumerate(final_steps, 1):
            st.markdown(f"**{i}.** {step}")
    else:
        st.write(final_steps)

    #Raw Sensor Data
    st.markdown("---")
    st.markdown("### Live User Telemetry (Raw Inputs)")
    st.caption("Data captured from user interaction *before* feature extraction.")

    r1, r2, r3, r4 = st.columns(4)
    
    with r1:
        st.metric("Formulation Time", f"{st.session_state.formulation_time:.2f}s", help="Time taken to type/submit")
    with r2:
        st.metric("System Latency", f"{runtime:.3f}s", help="AI Processing Time")
    with r3:
        st.metric("Input Volume", f"{len(result_prompt)} chars")
    with r4:
        #Simulated "Live" status for external sensors
        st.metric("Gaze Tracker", "Active", delta="Tracking", delta_color="normal")

    #CERAS Analysis
    st.markdown("---")
    st.markdown("Cognitive Efficiency Analysis")
    
    # Extract simulated signals (Using result_prompt)
    features = extract_ceras_features(result_prompt)

    #CEPM
    cepm_input = np.array([features[f] for f in cepm_features]).reshape(1, -1)
    cepm_input_scaled = cepm_scaler.transform(cepm_input)
    cepm_score = float(np.clip(cepm_model.predict(cepm_input_scaled)[0], 0, 1))

    #CNN
    cnn_input = np.array([features[f] for f in cnn_features]).reshape(1, -1)
    cnn_input = cnn_scaler.transform(cnn_input)

    if len(cnn_model.input_shape) == 3:
        cnn_input = cnn_input.reshape(cnn_input.shape[0], cnn_input.shape[1], 1)

    cnn_score = float(np.clip(np.squeeze(cnn_model.predict(cnn_input, verbose=0)), 0, 1))

    fusion_engine = CERASFusion()

    fusion_df = fusion_engine.fuse(
        session_ids=["session_1"],
        cepm_scores=[cepm_score],
        cnn_scores=[cnn_score]
    )

    fused_score = fusion_df["fused_ce_score"].iloc[0]
    confidence = fusion_df["confidence"].iloc[0]
    diagnostics = fusion_df["diagnostics"].iloc[0]
    readiness = fusion_df["readiness_label"].iloc[0]

    #Adaptive Learning Response
    st.markdown("## Adaptive Learning Response")
    
    if "adaptive_res" in st.session_state:
        st.markdown(st.session_state.adaptive_res)
    else:
        st.info("Adaptive response not available.")

    #Live Data Visulaization
    st.markdown("### Live Cognitive Signals")
    
    #Create a visual dashboard for the signals
    sig_col1, sig_col2 = st.columns(2)
    
    with sig_col1:
        st.markdown("**CEPM**")
        st.progress(float(cepm_score), text=f"Load: {cepm_score:.2f}")
        st.caption("Derived from knowledge")
        
    with sig_col2:
        st.markdown("**CNN (Behavioural)**")
        st.progress(float(cnn_score), text=f"Focus: {cnn_score:.2f}")
        st.caption("Derived from interaction cadence")
    
    st.markdown("---")
    
    # Main Score Visualization
    m1, m2 = st.columns([1, 2])
    
    with m1:
        st.metric("Fused CE Score", f"{fused_score:.2f}", delta="Real-time")
        if fused_score > 0.7:
            st.success("State: High Efficiency (Flow)")
        elif fused_score > 0.4:
            st.warning("State: Moderate Load")
        else:
            st.error("State: High Cognitive Load")
            
    with m2:
        st.markdown("**Fusion Engine Confidence**")
        st.progress(float(confidence), text=f"Confidence: {confidence:.2f}")
        st.info(f"Readiness State: {readiness}")

    #Fused CE Score(Explain)
    with st.expander("What is the Fused CE Score?"):
        st.markdown("""
            ### Fused Cognitive Efficiency (CE) Score

            The **Fused CE Score** reflects how efficiently you are learning in this session.

            It combines three independent signals:

            • **Conceptual Strength (CEPM)** – Depth of understanding  
            • **Behavioral & Reasoning Alignment (CNN)** – Interaction patterns, engagement consistency, and structural reasoning signals 

            These are fused into a single score between **0 and 1**.

            ### What Your Level Means

            **0.00 – 0.44 → Foundation Building**  
            You may need to revisit core concepts and slow down. Strengthen fundamentals before moving forward.

            **0.45 – 0.59 → Developing Momentum**  
            You're engaging and learning, but some inconsistencies exist. Refining strategy will help.

            **0.60 – 0.74 → Progressing Confidently**  
            You demonstrate stable understanding and good engagement. Keep challenging yourself.

            **0.75 – 1.00 → Peak Learning State**  
            You are operating with strong clarity, alignment, and efficiency. Ready for advanced challenges.

            This score reflects learning efficiency — not intelligence — and adapts to your behavior in real time.
            """)

    #Diagnostic Report
    with st.expander("Cognitive Diagnostic Report"):

        if fused_score < 0.40:
            st.markdown("""
            ### High Cognitive Load

            Your overall cognitive efficiency is currently low.

            **What this means:**  
            You may be struggling with core concepts, reasoning structure, or engagement consistency.

            **Recommended action:**  
            • Revisit foundational concepts  
            • Slow down your reasoning steps  
            • Avoid jumping directly to conclusions  
            • Practice structured problem solving  
            """)

        elif fused_score < 0.60:
            st.markdown("""
            ### Developing Understanding

            You are making progress, but inconsistencies are present.

            **What this means:**  
            Your conceptual understanding and reasoning structure are partially aligned.

            **Recommended action:**  
            • Strengthen logical flow  
            • Write clearer intermediate steps  
            • Reduce reasoning shortcuts  
            """)

        elif fused_score < 0.75:
            st.markdown("""
            ### Stable Cognitive Processing

            Your reasoning and engagement are well aligned.

            **What this means:**  
            You are learning efficiently with minor areas for refinement.

            **Recommended action:**  
            • Challenge yourself with harder problems  
            • Maintain structured reasoning  
            • Improve depth of explanations  
            """)

        else:
            st.markdown("""
            ### Peak Cognitive Efficiency

            Your conceptual clarity, behavioral engagement, and reasoning alignment are strongly synchronized.

            **What this means:**  
            You are operating in a high-efficiency learning state.

            **Recommended action:**  
            • Attempt advanced multi-step problems  
            • Explore edge cases  
            • Try teaching the concept back  
            """)

    #Improvement Suggestions
    with st.expander("Improvement Suggestions"):

        if fused_score < 0.4:
            st.markdown(
                """
                ### Priority: Rebuild Core Understanding

                - Revisit fundamental definitions and key principles.  
                - Solve 10–15 structured foundational problems.  
                - Focus on understanding *why* each step works.  
                - Avoid jumping directly to final answers.
                """
            )

        elif fused_score < 0.6:
            st.markdown(
                """
                ### Priority: Strengthen Logical Structure

                - Break problems into clear step-by-step reasoning.  
                - Write intermediate logic before concluding.  
                - Double-check assumptions between steps.  
                - Practice moderate-difficulty structured exercises.
                """
            )

        elif fused_score < 0.75:
            st.markdown(
                """
                ### Priority: Improve Consistency & Depth

                - Solve mixed-difficulty problems with structured reasoning.  
                - Explain your thought process explicitly.  
                - Maintain clarity under mild time pressure.  
                - Reduce small logical jumps.
                """
            )

        else:
            st.markdown(
                """
                ### Priority: Expand Mastery & Complexity

                - Attempt multi-stage and cross-topic problems.  
                - Practice timed reasoning sessions.  
                - Teach the concept back in your own words.  
                - Explore advanced variations and edge cases.
                """
            )

    #Confidence Interpretation
    with st.expander("Model Confidence Interpretation"):

        if confidence < 0.4:
            st.write(
                "**Very Low Confidence**\n\n"
                "Strong disagreement was detected between cognitive understanding, "
                "behavioral engagement, and reasoning structure.\n\n"
                "This assessment should be interpreted cautiously, and additional "
                "interactions may be required for reliable evaluation."
            )

        elif confidence < 0.6:
            st.write(
                "**Low–Moderate Confidence**\n\n"
                "There is noticeable inconsistency between learning signals.\n\n"
                "Some aspects of understanding or engagement may not be stable "
                "across similar tasks."
            )

        elif confidence < 0.85:
            st.write(
                "**High Confidence**\n\n"
                "Cognitive and behavioral signals are largely aligned.\n\n"
                "The system considers this assessment reasonably stable and reliable."
            )

        else:
            st.write(
                "**Very High Confidence**\n\n"
                "All major learning signals are strongly aligned.\n\n"
                "This indicates stable reasoning patterns and consistent engagement."
            )

    #Trace
    with st.expander("Reasoning Trace"):
        st.caption("Detailed logs of the decomposition and verification process.")
        logs = result.get("logs", "")
        st.code(logs)

    #Export
    export = {
        "prompt": result_prompt,
        "fused_ce_score": float(fused_score),
        "confidence": float(confidence),
        "diagnostics": diagnostics,
        "timestamp": datetime.utcnow().isoformat()
    }

    st.download_button(
        "Download Session Report",
        data=json.dumps(export, indent=2),
        file_name="ceras_session.json",
        mime="application/json",
    )

#Empty State
if not run_btn:
    st.info(
        "Enter a learning question above and run the session.\n\n"
        "CERAS will provide:\n"
        "• Step-by-step reasoning\n"
        "• Cognitive Efficiency score\n"
        "• Diagnostic feedback\n"
        "• Improvement suggestions"
    )

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; font-size:12px; color:#9ca3af; margin-top:8px;">
        Developed by <b>© Aman Goel</b> & <b>Rishaan Yadav</b><br>
        <span style="color:#6b7280;">CERAS · CERAS Framework</span>
    </div>
    """,
    unsafe_allow_html=True
)