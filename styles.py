# styles.py
import streamlit as st

def apply_style():
    st.markdown(
        """
        <style>
        /* Hide default Streamlit UI */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}

        .block-container { padding-top:0.1rem !important; padding-bottom: 0rem !important; padding-left:0.9rem !important; padding-right:0.9rem !important; }
        
        /* Aggressive spacing reduction between navbar and content */
        .main .block-container { padding-top: 0rem !important; margin-top: -1rem !important; }
        
        /* Target the space between navbar and first content element */
        .stApp > div:first-child > div:first-child > div:first-child > div:first-child { margin-top: -1rem !important; }
        
        /* Target the specific inference page content */
        .stApp > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child { margin-top: -0.5rem !important; }
        
        /* More targeted approach - only reduce space between navbar and first content */
        .stApp > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child { margin-top: -0.3rem !important; }
        
        /* Target only the first markdown element after navbar */
        .stApp > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child { margin-top: -0.2rem !important; }
        
        /* Ensure header position is maintained */
        .card-header { position: relative !important; top: 0 !important; }
        .card { position: relative !important; top: 0 !important; }
        
        /* Style for inline header with controls */
        .card-header { margin-bottom: 0 !important; }
        
        /* Align controls with header */
        .stDateInput, .stTextInput, .stSelectbox { margin-top: 0 !important; }
        .stDateInput > div, .stTextInput > div, .stSelectbox > div { margin-top: 0 !important; }
        
        /* Ensure controls are vertically centered with header */
        .stDateInput label, .stTextInput label, .stSelectbox label { margin-bottom: 0 !important; }
        .stDateInput > div > div, .stTextInput > div > div, .stSelectbox > div > div { margin-top: 0 !important; }
        
        /* Make all control labels have consistent font styling */
        .stDateInput label, .stTextInput label, .stSelectbox label { 
            font-size: 12px !important; 
            color: #94a3b8 !important; 
            margin-bottom: 2px !important; 
            font-weight: 500 !important;
        }
        
        /* Reduce spacing in inference page specifically */
        .stApp > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child { margin-top: -0.5rem !important; }
        
        /* Aggressively reduce spacing between table headers and data rows */
        .stMarkdown { margin-bottom: 0 !important; margin-top: 0 !important; }
        .stMarkdown > div { margin-bottom: 0 !important; margin-top: 0 !important; }
        .stMarkdown > div > div { margin-bottom: 0 !important; margin-top: 0 !important; }
        .stMarkdown > div > div > div { margin-bottom: 0 !important; margin-top: 0 !important; }
        
        /* Target all Streamlit elements in the table area */
        .stApp > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child { margin-top: -1rem !important; margin-bottom: 0 !important; }
        .stApp > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child { margin-top: -0.5rem !important; margin-bottom: 0 !important; }
        
        /* Reduce spacing in columns aggressively */
        .stColumns { margin-bottom: 0 !important; margin-top: 0 !important; }
        .stColumns > div { margin-bottom: 0 !important; margin-top: 0 !important; }
        .stColumns > div > div { margin-bottom: 0 !important; margin-top: 0 !important; }
        .stColumns > div > div > div { margin-bottom: 0 !important; margin-top: 0 !important; }
        
        /* Target specific table elements */
        .stApp > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child { margin-top: -0.3rem !important; margin-bottom: 0 !important; }
        
        /* Ultra-aggressive spacing reduction for table area */
        .stApp > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child { margin-top: -2rem !important; margin-bottom: 0 !important; }
        .stApp > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child { margin-top: -1rem !important; margin-bottom: 0 !important; }
        
        /* Target all elements in the table section */
        .stApp > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child { margin-top: -0.5rem !important; margin-bottom: 0 !important; }
        
        /* Reduce height of Email and SMS buttons */
        .stButton > button { 
            height: 28px !important; 
            min-height: 28px !important;
            padding: 4px 8px !important;
            font-size: 12px !important;
            line-height: 1.2 !important;
        }
        
        /* Target buttons in the table specifically */
        .stApp > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child .stButton > button {
            height: 24px !important;
            min-height: 24px !important;
            padding: 2px 6px !important;
            font-size: 11px !important;
        }

        .navbar {
            position: sticky; top:0; z-index:9999; display:flex; align-items:center; justify-content:space-between;
            padding:12px 16px; background:#ffffff; box-shadow:0 2px 6px rgba(16,24,40,0.06); margin-bottom:2px;
        }
        .navbar::before { content:""; position:absolute; top:0; left:0; right:0; height:3px; background:linear-gradient(90deg,#c3002f,#000000); }
        .navbar::after { content:""; position:absolute; bottom:0; left:0; right:0; height:3px; background:linear-gradient(90deg,#000000,#c3002f); }
        .navbar .title { font-weight:600; font-size:16px; color:#111827; }
        .title-column { display:flex; flex-direction:column; gap:2px; }
        .subtitle {
            font-size:12px;
            color:#6b7280;            /* slightly muted gray */
            font-weight:500;
            margin-top:2px;
            letter-spacing:0.1px;
        }

        .card { background:#0b0f13; border-radius:12px; padding:8px 10px; margin-bottom:8px; border:1px solid rgba(255,255,255,0.03); box-shadow:0 2px 6px rgba(0,0,0,0.6); }
        .card-header { font-weight:600; font-size:14px; color:#e6eef8; margin-bottom:8px; background:rgba(255,255,255,0.03); padding:8px 12px; border-radius:10px; }

        /* KPI tweaks */
        .kpi-wrap { position:relative; display:inline-block; }
        .kpi-label { font-size:13px; color:#cbd5e1; margin-bottom:2px; white-space:nowrap; } /* prevent wrapping */
        .kpi-num {
            font-size:28px;
            font-weight:700;
            color:#ffffff;
            line-height:0.95;
            text-shadow: 0 1px 0 rgba(255,255,255,0.03), 0 0 8px rgba(0,0,0,0.6);
            opacity: 0;
            animation: fadeInKPI 0.5s ease forwards;
        }
        .kpi-tooltip {
            visibility:hidden; width:220px; background:rgba(17,24,39,0.95); color:#fff; text-align:center; border-radius:6px; padding:8px; position:absolute; z-index:99999; bottom:110%; left:50%; transform:translateX(-50%); opacity:0; transition:opacity 0.18s ease-in-out; box-shadow:0 6px 18px rgba(2,6,23,0.6); font-size:13px;
        }
        .kpi-wrap:hover .kpi-tooltip, .kpi-claim-wrap:hover .kpi-tooltip { visibility:visible; opacity:1; }

        @keyframes fadeInKPI {
            from { opacity: 0; transform: translateY(6px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        /* Center align incidents/claims values */
        .stat-centered { text-align:center; }
        .stat-label { font-size:12px; color:#94a3b8; margin-bottom:0px; }
        .stat-value { font-size:18px; font-weight:700; color:#e6eef8; }

        .stSelectbox > div[data-baseweb="select"] { min-height:40px; }
        .stSelectbox label { font-size:12px; color:#94a3b8; margin-bottom:2px; }

        @media (max-width:900px) {
            .kpi-num { font-size:34px; }
        }

        .chart-header {
            font-size:15px;
            font-weight:600;
            color:#94a3b8;
            margin: 4px 0 6px 2px;
        }

        /* Predictive Info pulse animation */
        .pulse {
            animation: pulseAnim 1s ease-in-out;
        }
        @keyframes pulseAnim {
            0% { box-shadow: 0 0 0 rgba(195,0,47,0.0); }
            30% { box-shadow: 0 0 15px rgba(195,0,47,0.7); }
            60% { box-shadow: 0 0 15px rgba(195,0,47,0.5); }
            100% { box-shadow: 0 0 0 rgba(195,0,47,0.0); }
        }

        /* Compact slider styling */
        div[data-testid="stSelectSlider"] {
            padding: 0 !important;
            margin: -4px 0 0 0 !important;
        }
        div[data-testid="stSelectSlider"] > div {
            font-size: 11px !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )
