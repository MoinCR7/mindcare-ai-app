
# ============================
# 1. Imports
# ============================
import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import numpy as np
import pandas as pd # <-- Added pandas for data display
import speech_recognition as sr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============================
# 2. Page Configuration
# ============================
st.set_page_config(
    page_title="mindcare.ai - Your AI Wellness Companion",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================
# 3. Custom CSS for High-Fidelity UI
# ============================
st.markdown("""
<style>
    /* Main container and block styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    /* --- Login Page Styling --- */
    .login-container {
        background-color: #F0F2F6;
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    
    /* --- Modern Arcade/Flashcard Button Styling --- */
    .card {
        background-color: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: all 0.3s ease-in-out;
        border: 1px solid #E0E0E0;
        text-align: center;
        height: 100%; /* Make cards in a row equal height */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .card:hover {
        transform: translateY(-10px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }
    .card-emoji {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .card-description {
        font-size: 1rem;
        color: #666;
        flex-grow: 1; /* Pushes button to the bottom */
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        padding: 10px 0;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# 4. Model & Artifact Loading (Cached)
# ============================
@st.cache_resource
def load_artifacts():
    try:
        emotion_model = load_model("artifacts/emotion_model.keras")
        sev_model = load_model("artifacts/severity_model.keras")
        tokenizer = joblib.load("artifacts/tokenizer.pkl")
        label_enc = joblib.load("artifacts/label_encoder.pkl")
        sev_enc = joblib.load("artifacts/severity_encoder.pkl")
        return emotion_model, sev_model, tokenizer, label_enc, sev_enc
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None, None, None

emotion_model, sev_model, tokenizer, label_enc, sev_enc = load_artifacts()

EMOJI_MAP = {"anger": "😠", "fear": "😨", "joy": "😄", "love": "❤️", "sadness": "😢", "surprise": "😮"}

# ============================
# 5. Core Functions (Prediction & Voice)
# ============================

# MODIFIED: predict() now returns probabilities
def predict(text, maxlen=60):
    if not all([emotion_model, sev_model, tokenizer, label_enc, sev_enc]): 
        return "Error", "Models not loaded", "Error", None, None
    
    # Preprocess the text
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")

    # Predict emotion and get probabilities
    pred_em_probs = emotion_model.predict(padded_seq)[0]
    pred_em_idx = np.argmax(pred_em_probs)
    emotion = label_enc.classes_[pred_em_idx]

    # Predict severity and get probabilities
    pred_sev_probs = sev_model.predict(padded_seq)[0]
    pred_sev_idx = np.argmax(pred_sev_probs)
    severity = sev_enc.classes_[pred_sev_idx]

    emoji = EMOJI_MAP.get(emotion, "❓")
    
    return emotion, severity, emoji, pred_em_probs, pred_sev_probs

# (voice_to_text function remains the same)
def voice_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now.")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            text = r.recognize_google(audio)
            st.success("Voice recognized!")
            return text
        except Exception:
            st.error("Sorry, I could not understand the audio.")
    return ""
    
# ============================
# 6. UI Pages
# ============================

# (Session state and navigation functions remain the same)
if 'page' not in st.session_state: st.session_state.page = 'Login'
if 'user_input' not in st.session_state: st.session_state.user_input = ""
if 'selected_section' not in st.session_state: st.session_state.selected_section = "Mood Check-in"

def navigate_to(page_name): st.session_state.page = page_name
def select_section(section_name):
    st.session_state.selected_section = section_name
    st.session_state.page = 'MainApp'
    
# (page_login and page_selection_screen remain the same)
def page_login():
    st.title("Welcome back to mindcare.ai 🧠")
    st.markdown("Your personal space for mental wellness.")
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        with st.container():
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            st.subheader("Sign In")
            st.text_input("Username or Email", placeholder="Enter anything")
            st.text_input("Password", type="password", placeholder="Enter anything")
            if st.button("Sign In", use_container_width=True, type="primary"):
                navigate_to('SelectionScreen')
                st.rerun()

            st.divider()

            if st.button("Continue as Guest", use_container_width=True):
                navigate_to('SelectionScreen')
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

def page_selection_screen():
    st.title("How can I help you today?")
    st.write("")
    
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        with st.container():
            st.markdown("""
            <div class="card">
                <div class="card-emoji">📝</div>
                <div class="card-title">Mood Check-in</div>
                <div class="card-description">Analyze your current feelings through text or voice. Get instant insights into your emotional state.</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Start Check-in", key="checkin", use_container_width=True):
                select_section("Mood Check-in")
                st.rerun()
                
    with col2:
        with st.container():
            st.markdown("""
            <div class="card">
                <div class="card-emoji">🛠️</div>
                <div class="card-title">Wellness Toolkit</div>
                <div class="card-description">Access a curated set of tools like guided meditations, journaling prompts, and breathing exercises.</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Open Toolkit", key="tools", use_container_width=True):
                select_section("Tools & Resources")
                st.rerun()

    with col3:
        with st.container():
            st.markdown("""
            <div class="card">
                <div class="card-emoji">🧑‍⚕️</div>
                <div class="card-title">Therapy Hub</div>
                <div class="card-description">Explore options for professional support, find resources, or connect with our future AI therapy chatbot.</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Visit Hub", key="hub", use_container_width=True):
                select_section("Therapy Hub")
                st.rerun()

# --- 3. Main Application Page with Sidebar ---
def page_main_app():
    sections = ["Mood Check-in", "Dashboard", "Tools & Resources", "Therapy Hub", "Gamify"]
    try:
        default_idx = sections.index(st.session_state.selected_section)
    except ValueError:
        default_idx = 0

    with st.sidebar:
        st.title("mindcare.ai")
        selected = option_menu(
            "Navigation", sections,
            icons=['emoji-smile', 'bar-chart-line', 'tools', 'hospital', 'joystick'],
            menu_icon="cast", default_index=default_idx,
            styles={"nav-link-selected": {"background-color": st.get_option("theme.primaryColor")}},
        )
        st.divider()
        if st.button("Back to Main Menu"):
             navigate_to('SelectionScreen')
             st.rerun()

    # --- Page 1: Mood Check-in (The Core Feature) ---
    if selected == "Mood Check-in":
        st.title("How are you feeling today?")
        st.markdown("Use the space below to write down what's on your mind. The more you write, the better I can understand.")

        col1, col2 = st.columns([3, 1])
        with col1:
            user_text = st.text_area("Type your thoughts here...", height=150, key="user_input_area", value=st.session_state.user_input)
        with col2:
            st.write("")
            st.write("")
            if st.button("Use Voice to Text 🎤", use_container_width=True):
                text_from_voice = voice_to_text()
                if text_from_voice:
                    st.session_state.user_input = text_from_voice
                    st.rerun()
            
            analyze_button = st.button("Analyze My Mood ✨", use_container_width=True, type="primary")

        # MODIFIED: Updated results section to display probabilities
        if analyze_button and user_text:
            with st.spinner("Analyzing your feelings..."):
                emotion, severity, emoji, emotion_probs, severity_probs = predict(user_text)

            st.write("---")
            st.subheader("Analysis Complete!")
            
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.markdown(f"<h1 style='text-align: center; font-size: 80px;'>{emoji}</h1>", unsafe_allow_html=True)
            with res_col2:
                st.markdown(f"It sounds like you're feeling **{emotion.capitalize()}**.")
                st.markdown(f"The intensity of this feeling seems to be **{severity}**.")

            st.write("")
            with st.expander("🔍 View Prediction Details"):
                st.markdown("Here's the probability distribution for your input. This shows how confident the model is about each category.")
                
                # Create DataFrames for display
                emotion_df = pd.DataFrame({
                    'Emotion': label_enc.classes_,
                    'Probability': emotion_probs
                }).sort_values(by='Probability', ascending=False).reset_index(drop=True)

                severity_df = pd.DataFrame({
                    'Severity': sev_enc.classes_,
                    'Probability': severity_probs
                }).sort_values(by='Probability', ascending=False).reset_index(drop=True)

                # Style the DataFrames
                st.subheader("Emotion Probabilities")
                st.dataframe(emotion_df.style.format({'Probability': '{:.2%}'}).background_gradient(cmap='Greens', subset=['Probability']), use_container_width=True)

                st.subheader("Severity Probabilities")
                st.dataframe(severity_df.style.format({'Probability': '{:.2%}'}).background_gradient(cmap='Oranges', subset=['Probability']), use_container_width=True)

                st.info("A model favoring one prediction (e.g., 'anger') might show slightly higher probabilities for it even when other emotions are also likely. This is often due to the data it was trained on.", icon="💡")

    # (Placeholder pages remain the same)
    elif selected in ["Dashboard", "Tools & Resources", "Therapy Hub", "Gamify"]:
        st.title(selected)
        st.info("This section is under development. Stay tuned for updates!", icon="🛠️")
        st.image("https://static.streamlit.io/examples/cat.jpg", caption="Enjoy this placeholder picture!")


# ============================
# 7. Main App Router
# ============================
if not emotion_model:
    st.error("Application cannot start. Please check the logs for errors in loading models.")
else:
    if st.session_state.page == 'Login': page_login()
    elif st.session_state.page == 'SelectionScreen': page_selection_screen()
=======
# ============================
# 1. Imports
# ============================
import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import numpy as np
import pandas as pd # <-- Added pandas for data display
import speech_recognition as sr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============================
# 2. Page Configuration
# ============================
st.set_page_config(
    page_title="mindcare.ai - Your AI Wellness Companion",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================
# 3. Custom CSS for High-Fidelity UI
# ============================
st.markdown("""
<style>
    /* Main container and block styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    /* --- Login Page Styling --- */
    .login-container {
        background-color: #F0F2F6;
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    
    /* --- Modern Arcade/Flashcard Button Styling --- */
    .card {
        background-color: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: all 0.3s ease-in-out;
        border: 1px solid #E0E0E0;
        text-align: center;
        height: 100%; /* Make cards in a row equal height */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .card:hover {
        transform: translateY(-10px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }
    .card-emoji {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .card-description {
        font-size: 1rem;
        color: #666;
        flex-grow: 1; /* Pushes button to the bottom */
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        padding: 10px 0;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# 4. Model & Artifact Loading (Cached)
# ============================
@st.cache_resource
def load_artifacts():
    try:
        emotion_model = load_model("artifacts/emotion_model.keras")
        sev_model = load_model("artifacts/severity_model.keras")
        tokenizer = joblib.load("artifacts/tokenizer.pkl")
        label_enc = joblib.load("artifacts/label_encoder.pkl")
        sev_enc = joblib.load("artifacts/severity_encoder.pkl")
        return emotion_model, sev_model, tokenizer, label_enc, sev_enc
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None, None, None

emotion_model, sev_model, tokenizer, label_enc, sev_enc = load_artifacts()

EMOJI_MAP = {"anger": "😠", "fear": "😨", "joy": "😄", "love": "❤️", "sadness": "😢", "surprise": "😮"}

# ============================
# 5. Core Functions (Prediction & Voice)
# ============================

# MODIFIED: predict() now returns probabilities
def predict(text, maxlen=60):
    if not all([emotion_model, sev_model, tokenizer, label_enc, sev_enc]): 
        return "Error", "Models not loaded", "Error", None, None
    
    # Preprocess the text
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")

    # Predict emotion and get probabilities
    pred_em_probs = emotion_model.predict(padded_seq)[0]
    pred_em_idx = np.argmax(pred_em_probs)
    emotion = label_enc.classes_[pred_em_idx]

    # Predict severity and get probabilities
    pred_sev_probs = sev_model.predict(padded_seq)[0]
    pred_sev_idx = np.argmax(pred_sev_probs)
    severity = sev_enc.classes_[pred_sev_idx]

    emoji = EMOJI_MAP.get(emotion, "❓")
    
    return emotion, severity, emoji, pred_em_probs, pred_sev_probs

# (voice_to_text function remains the same)
def voice_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now.")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            text = r.recognize_google(audio)
            st.success("Voice recognized!")
            return text
        except Exception:
            st.error("Sorry, I could not understand the audio.")
    return ""
    
# ============================
# 6. UI Pages
# ============================

# (Session state and navigation functions remain the same)
if 'page' not in st.session_state: st.session_state.page = 'Login'
if 'user_input' not in st.session_state: st.session_state.user_input = ""
if 'selected_section' not in st.session_state: st.session_state.selected_section = "Mood Check-in"

def navigate_to(page_name): st.session_state.page = page_name
def select_section(section_name):
    st.session_state.selected_section = section_name
    st.session_state.page = 'MainApp'
    
# (page_login and page_selection_screen remain the same)
def page_login():
    st.title("Welcome back to mindcare.ai 🧠")
    st.markdown("Your personal space for mental wellness.")
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        with st.container():
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            st.subheader("Sign In")
            st.text_input("Username or Email", placeholder="Enter anything")
            st.text_input("Password", type="password", placeholder="Enter anything")
            if st.button("Sign In", use_container_width=True, type="primary"):
                navigate_to('SelectionScreen')
                st.rerun()

            st.divider()

            if st.button("Continue as Guest", use_container_width=True):
                navigate_to('SelectionScreen')
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

def page_selection_screen():
    st.title("How can I help you today?")
    st.write("")
    
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        with st.container():
            st.markdown("""
            <div class="card">
                <div class="card-emoji">📝</div>
                <div class="card-title">Mood Check-in</div>
                <div class="card-description">Analyze your current feelings through text or voice. Get instant insights into your emotional state.</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Start Check-in", key="checkin", use_container_width=True):
                select_section("Mood Check-in")
                st.rerun()
                
    with col2:
        with st.container():
            st.markdown("""
            <div class="card">
                <div class="card-emoji">🛠️</div>
                <div class="card-title">Wellness Toolkit</div>
                <div class="card-description">Access a curated set of tools like guided meditations, journaling prompts, and breathing exercises.</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Open Toolkit", key="tools", use_container_width=True):
                select_section("Tools & Resources")
                st.rerun()

    with col3:
        with st.container():
            st.markdown("""
            <div class="card">
                <div class="card-emoji">🧑‍⚕️</div>
                <div class="card-title">Therapy Hub</div>
                <div class="card-description">Explore options for professional support, find resources, or connect with our future AI therapy chatbot.</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Visit Hub", key="hub", use_container_width=True):
                select_section("Therapy Hub")
                st.rerun()

# --- 3. Main Application Page with Sidebar ---
def page_main_app():
    sections = ["Mood Check-in", "Dashboard", "Tools & Resources", "Therapy Hub", "Gamify"]
    try:
        default_idx = sections.index(st.session_state.selected_section)
    except ValueError:
        default_idx = 0

    with st.sidebar:
        st.title("mindcare.ai")
        selected = option_menu(
            "Navigation", sections,
            icons=['emoji-smile', 'bar-chart-line', 'tools', 'hospital', 'joystick'],
            menu_icon="cast", default_index=default_idx,
            styles={"nav-link-selected": {"background-color": st.get_option("theme.primaryColor")}},
        )
        st.divider()
        if st.button("Back to Main Menu"):
             navigate_to('SelectionScreen')
             st.rerun()

    # --- Page 1: Mood Check-in (The Core Feature) ---
    if selected == "Mood Check-in":
        st.title("How are you feeling today?")
        st.markdown("Use the space below to write down what's on your mind. The more you write, the better I can understand.")

        col1, col2 = st.columns([3, 1])
        with col1:
            user_text = st.text_area("Type your thoughts here...", height=150, key="user_input_area", value=st.session_state.user_input)
        with col2:
            st.write("")
            st.write("")
            if st.button("Use Voice to Text 🎤", use_container_width=True):
                text_from_voice = voice_to_text()
                if text_from_voice:
                    st.session_state.user_input = text_from_voice
                    st.rerun()
            
            analyze_button = st.button("Analyze My Mood ✨", use_container_width=True, type="primary")

        # MODIFIED: Updated results section to display probabilities
        if analyze_button and user_text:
            with st.spinner("Analyzing your feelings..."):
                emotion, severity, emoji, emotion_probs, severity_probs = predict(user_text)

            st.write("---")
            st.subheader("Analysis Complete!")
            
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.markdown(f"<h1 style='text-align: center; font-size: 80px;'>{emoji}</h1>", unsafe_allow_html=True)
            with res_col2:
                st.markdown(f"It sounds like you're feeling **{emotion.capitalize()}**.")
                st.markdown(f"The intensity of this feeling seems to be **{severity}**.")

            st.write("")
            with st.expander("🔍 View Prediction Details"):
                st.markdown("Here's the probability distribution for your input. This shows how confident the model is about each category.")
                
                # Create DataFrames for display
                emotion_df = pd.DataFrame({
                    'Emotion': label_enc.classes_,
                    'Probability': emotion_probs
                }).sort_values(by='Probability', ascending=False).reset_index(drop=True)

                severity_df = pd.DataFrame({
                    'Severity': sev_enc.classes_,
                    'Probability': severity_probs
                }).sort_values(by='Probability', ascending=False).reset_index(drop=True)

                # Style the DataFrames
                st.subheader("Emotion Probabilities")
                st.dataframe(emotion_df.style.format({'Probability': '{:.2%}'}).background_gradient(cmap='Greens', subset=['Probability']), use_container_width=True)

                st.subheader("Severity Probabilities")
                st.dataframe(severity_df.style.format({'Probability': '{:.2%}'}).background_gradient(cmap='Oranges', subset=['Probability']), use_container_width=True)

                st.info("A model favoring one prediction (e.g., 'anger') might show slightly higher probabilities for it even when other emotions are also likely. This is often due to the data it was trained on.", icon="💡")

    # (Placeholder pages remain the same)
    elif selected in ["Dashboard", "Tools & Resources", "Therapy Hub", "Gamify"]:
        st.title(selected)
        st.info("This section is under development. Stay tuned for updates!", icon="🛠️")
        st.image("https://static.streamlit.io/examples/cat.jpg", caption="Enjoy this placeholder picture!")


# ============================
# 7. Main App Router
# ============================
if not emotion_model:
    st.error("Application cannot start. Please check the logs for errors in loading models.")
else:
    if st.session_state.page == 'Login': page_login()
    elif st.session_state.page == 'SelectionScreen': page_selection_screen()
>>>>>>> ec951ac24cc80efce6e9f051ff4761386386c8a9
    elif st.session_state.page == 'MainApp': page_main_app()
