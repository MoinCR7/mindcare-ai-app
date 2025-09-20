import streamlit as st

# --- PAGE CONFIGURATION ---
# Must be the first Streamlit command.
st.set_page_config(
    page_title="MindCare AI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- ASSET URLS ---
LOGIN_BG_URL = "https://images.pexels.com/photos/3293029/pexels-photo-3293029.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
LOGO_URL = "https://i.imgur.com/264422w.png"

# --- STYLES ---
# All CSS is now embedded in this function for cleaner code.
def load_styles():
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
        body {{
            font-family: 'Poppins', sans-serif;
        }}

        /* --- Hides Streamlit's default elements --- */
        #MainMenu, footer, .stDeployButton {{
            display: none;
        }}

        /* --- Login page specific styles --- */
        .login-page-container {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-image: url({LOGIN_BG_URL});
            background-size: cover;
            background-position: center;
        }}
        .glass-container {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100%;
        }}
        .glass-form {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            padding: 40px;
            width: 90%;
            max-width: 420px;
            text-align: center;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }}
        .glass-form h2 {{
            margin-bottom: 20px;
            color: white;
            font-weight: 700;
        }}

        /* --- Logged-in Home page styles --- */
        .home-card {{
            background-color: #F0F2F6;
            border-radius: 12px;
            padding: 25px;
            text-align: center;
            transition: all 0.3s ease-in-out;
            border: 1px solid #E0E0E0;
        }}
        .home-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.1);
        }}
        .home-card a {{
            text-decoration: none;
            color: #333;
        }}
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = "Guest"

# =====================================================================
#                          MAIN APP LOGIC
# =====================================================================

load_styles()

# --- 1. IF USER IS LOGGED IN, SHOW THE HOME/DASHBOARD PAGE ---
if st.session_state['logged_in']:
    
    # --- Sidebar ---
    # The sidebar is consistent across all logged-in pages.
    st.sidebar.image(LOGO_URL, width=70)
    st.sidebar.header(f"Welcome, {st.session_state['username']}!")
    st.sidebar.success("You are logged in. Select a page above.")

    # --- Home Page Content ---
    st.title(f"Welcome to MindCare AI, {st.session_state['username']}!")
    st.markdown("Your personal companion for understanding and improving your mental well-being. What would you like to do today?")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        # Use st.page_link for robust navigation
        st.page_link(
            "pages/1_Analyzer.py",
            label="""
            <div class="home-card">
                <h3>🧠</h3>
                <h4>Analyze Your Mood</h4>
                <p>Check in and understand your current emotional state.</p>
            </div>""",
            unsafe_allow_html=True
        )

    with col2:
        st.page_link(
            "pages/2_Resources.py",
            label="""
            <div class="home-card">
                <h3>📚</h3>
                <h4>Explore Resources</h4>
                <p>Access tools like meditations and journaling.</p>
            </div>""",
            unsafe_allow_html=True
        )
    
    st.page_link(
        "pages/3_Support_Achievements.py",
        label="""
        <div class="home-card">
            <h3>🏆</h3>
            <h4>Get Support & See Progress</h4>
            <p>Find professional help and track your achievements.</p>
        </div>""",
        unsafe_allow_html=True
    )


# --- 2. IF USER IS NOT LOGGED IN, SHOW THE LOGIN PAGE ---
else:
    # This injects the background and the glassmorphism container
    st.markdown('<div class="login-page-container">', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="glass-container"><div class="glass-form">', unsafe_allow_html=True)
        
        st.image(LOGO_URL, width=100)
        st.markdown("<h2>MindCare AI</h2>", unsafe_allow_html=True)
        
        username = st.text_input(
            "Enter your name to begin", 
            placeholder="e.g., Alex", 
            label_visibility="collapsed"
        )
        
        if st.button("Continue", use_container_width=True, type="primary"):
            if username:
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.rerun()
            else:
                st.warning("Please enter your name.")

        st.markdown('</div></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
