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

        .navbar {
            position: sticky; top:0; z-index:9999; display:flex; align-items:center; justify-content:space-between;
            padding:12px 16px; background:#ffffff; box-shadow:0 2px 6px rgba(16,24,40,0.06); margin-bottom:8px;
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

        .card { background:#0b0f13; border-radius:8px; padding:8px 10px; margin-bottom:8px; border:1px solid rgba(255,255,255,0.03); box-shadow:0 2px 6px rgba(0,0,0,0.6); }
        .card-header { font-weight:600; font-size:14px; color:#e6eef8; margin-bottom:8px; background:rgba(255,255,255,0.03); padding:8px 12px; border-radius:6px; }

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
