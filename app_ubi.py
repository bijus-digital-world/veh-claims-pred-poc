import base64
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import requests
import json
import time
import random
import math

import streamlit as st
import pydeck as pdk


def set_page_config() -> None:
    # Use custom UBI icon
    icon_path = Path(__file__).parent / "images" / "ubi_app_icon.png"
    page_icon = str(icon_path) if icon_path.exists() else "🚗"
    
    st.set_page_config(
        page_title="UBI Analytics – Telematics",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="collapsed",
    )


def inject_js() -> None:
    st.markdown(
        """
        <script>
        // Run immediately and on DOM ready
        function fixSpacing() {
            // Remove all top spacing from Streamlit containers
            const selectors = [
                '.stApp',
                '.main',
                '.block-container',
                '[data-testid="stAppViewContainer"]',
                '[data-testid="stVerticalBlock"]',
                'div[data-testid="stVerticalBlock"] > div',
                '.main > div',
                '.main .block-container > div'
            ];
            
            selectors.forEach(selector => {
                const elements = document.querySelectorAll(selector);
                elements.forEach(el => {
                    el.style.paddingTop = '0px';
                    el.style.marginTop = '0px';
                });
            });
            
            // Force navbar to very top
            const navbar = document.querySelector('.top-nav');
            if (navbar) {
                navbar.style.marginTop = '2px';
                navbar.style.position = 'sticky';
                navbar.style.top = '0';
            }
        }
        
        // Run immediately
        fixSpacing();
        
        // Run when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', fixSpacing);
        } else {
            fixSpacing();
        }
        
        // Run after a short delay to catch any late-rendered elements
        setTimeout(fixSpacing, 100);
        </script>
        """,
        unsafe_allow_html=True,
    )

    # (map rendering occurs in render_showcase, not here)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-dark: #0a1018;
            --bg-darker: #050810;
            --text: #e6eefc;
            --muted: #9fb0d7;
            --card: rgba(17, 26, 44, 0.7);
            --card-border: rgba(90, 122, 190, 0.25);
            --primary-1: #60a5fa;   /* blue-400 */
            --primary-2: #2563eb;   /* blue-600 */
            --primary-3: #1e40af;   /* blue-800 */
            --accent: #22d3ee;      /* cyan-400 */
        }

        html, body { background: var(--bg-dark); color: var(--text); }
        .stApp { background: transparent !important; color: var(--text) !important; }

        /* Animated background - Strands-style floating orbs */
        .page-bg {
            position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            z-index: -2; pointer-events: none;
            background: var(--bg-dark);
            overflow: hidden;
        }
        
        /* Additional floating elements for more dynamic movement */
        .page-bg .float-extra-1 {
            content: ""; position: absolute; top: 10%; left: 10%; width: 150px; height: 150px;
            background: radial-gradient(circle, rgba(59,130,246,0.18), transparent 60%);
            border-radius: 50%;
            animation: floatExtra1 25s ease-in-out infinite;
        }
        .page-bg .float-extra-2 {
            content: ""; position: absolute; top: 70%; right: 15%; width: 120px; height: 120px;
            background: radial-gradient(circle, rgba(34,211,238,0.15), transparent 60%);
            border-radius: 50%;
            animation: floatExtra2 20s ease-in-out infinite reverse;
        }
        .page-bg .float-extra-3 {
            content: ""; position: absolute; top: 40%; left: 60%; width: 130px; height: 130px;
            background: radial-gradient(circle, rgba(147,197,253,0.12), transparent 60%);
            border-radius: 50%;
            animation: floatExtra3 28s ease-in-out infinite;
        }
        .page-bg::before {
            content: ""; position: absolute; top: -20%; left: -20%; width: 140%; height: 140%;
            background:
                radial-gradient(500px 400px at 20% 30%, rgba(37,99,235,0.35), transparent 60%),
                radial-gradient(400px 350px at 80% 20%, rgba(96,165,250,0.28), transparent 60%),
                radial-gradient(450px 380px at 60% 80%, rgba(34,211,238,0.22), transparent 60%),
                radial-gradient(350px 300px at 40% 10%, rgba(59,130,246,0.18), transparent 50%);
            animation: float1 18s ease-in-out infinite;
        }
        .page-bg::after {
            content: ""; position: absolute; top: -30%; left: -30%; width: 160%; height: 160%;
            background:
                radial-gradient(600px 500px at 70% 40%, rgba(59,130,246,0.28), transparent 65%),
                radial-gradient(500px 450px at 30% 70%, rgba(147,197,253,0.22), transparent 65%),
                radial-gradient(550px 480px at 50% 10%, rgba(14,165,233,0.18), transparent 65%),
                radial-gradient(400px 350px at 90% 60%, rgba(37,99,235,0.15), transparent 55%);
            animation: float2 22s ease-in-out infinite reverse;
        }
        /* Horizontal drifting elements like Strands */
        .page-bg .drift-left {
            content: ""; position: absolute; top: 20%; left: -150px; width: 250px; height: 250px;
            background: radial-gradient(circle, rgba(96,165,250,0.25), transparent 60%);
            border-radius: 50%;
            animation: driftLeft 30s linear infinite;
        }
        .page-bg .drift-right {
            content: ""; position: absolute; top: 60%; right: -150px; width: 200px; height: 200px;
            background: radial-gradient(circle, rgba(34,211,238,0.20), transparent 60%);
            border-radius: 50%;
            animation: driftRight 25s linear infinite;
        }
        @keyframes float1 {
            0% { transform: translate(0, 0) scale(1) rotate(0deg); }
            20% { transform: translate(60px, -40px) scale(1.25) rotate(72deg); }
            40% { transform: translate(-50px, 60px) scale(0.8) rotate(144deg); }
            60% { transform: translate(45px, 20px) scale(1.15) rotate(216deg); }
            80% { transform: translate(-30px, -35px) scale(0.9) rotate(288deg); }
            100% { transform: translate(0, 0) scale(1) rotate(360deg); }
        }
        @keyframes float2 {
            0% { transform: translate(0, 0) scale(1) rotate(0deg); }
            20% { transform: translate(-70px, 30px) scale(1.4) rotate(-72deg); }
            40% { transform: translate(40px, -60px) scale(0.7) rotate(-144deg); }
            60% { transform: translate(-25px, -30px) scale(1.25) rotate(-216deg); }
            80% { transform: translate(35px, 45px) scale(0.85) rotate(-288deg); }
            100% { transform: translate(0, 0) scale(1) rotate(-360deg); }
        }
        @keyframes driftLeft {
            0% { transform: translateX(-400px) translateY(0px) scale(0.7); }
            20% { transform: translateX(20vw) translateY(-60px) scale(1.3); }
            40% { transform: translateX(40vw) translateY(30px) scale(0.8); }
            60% { transform: translateX(60vw) translateY(-45px) scale(1.1); }
            80% { transform: translateX(80vw) translateY(25px) scale(0.9); }
            100% { transform: translateX(calc(100vw + 400px)) translateY(0px) scale(0.7); }
        }
        @keyframes driftRight {
            0% { transform: translateX(400px) translateY(0px) scale(0.7); }
            20% { transform: translateX(-20vw) translateY(50px) scale(1.2); }
            40% { transform: translateX(-40vw) translateY(-25px) scale(0.8); }
            60% { transform: translateX(-60vw) translateY(40px) scale(1.3); }
            80% { transform: translateX(-80vw) translateY(-20px) scale(0.9); }
            100% { transform: translateX(calc(-100vw - 400px)) translateY(0px) scale(0.7); }
        }
        
        /* Additional floating animations */
        @keyframes floatExtra1 {
            0%, 100% { transform: translate(0, 0) scale(1) rotate(0deg); }
            25% { transform: translate(80px, -60px) scale(1.4) rotate(90deg); }
            50% { transform: translate(-60px, 70px) scale(0.7) rotate(180deg); }
            75% { transform: translate(40px, -30px) scale(1.2) rotate(270deg); }
        }
        @keyframes floatExtra2 {
            0%, 100% { transform: translate(0, 0) scale(1) rotate(0deg); }
            30% { transform: translate(-70px, -50px) scale(1.3) rotate(108deg); }
            60% { transform: translate(50px, 40px) scale(0.8) rotate(216deg); }
            90% { transform: translate(-30px, 60px) scale(1.1) rotate(324deg); }
        }
        @keyframes floatExtra3 {
            0%, 100% { transform: translate(0, 0) scale(1) rotate(0deg); }
            20% { transform: translate(60px, 40px) scale(1.2) rotate(72deg); }
            40% { transform: translate(-40px, -70px) scale(0.8) rotate(144deg); }
            60% { transform: translate(70px, -30px) scale(1.3) rotate(216deg); }
            80% { transform: translate(-50px, 50px) scale(0.9) rotate(288deg); }
        }

        /* Hide Streamlit default chrome */
        div[data-testid="stDecoration"], div[data-testid="stToolbar"], div[data-testid="stHeader"], header[tabindex="-1"] {
            display: none !important; height: 0 !important; margin: 0 !important; padding: 0 !important; visibility: hidden !important;
        }

        /* Aggressively override Streamlit's default spacing */
        .stApp {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        
        .main { 
            padding-top: 0 !important; 
            margin-top: 0 !important;
        }
        
        .block-container { max-width: 1180px; padding-top:3rem !important; padding-bottom: 0rem !important; 
            padding-left:0.9rem !important; padding-right:0.9rem !important; position: relative; 
            z-index: 1; background: transparent !important;}

        /*.main .block-container { 
            max-width: 1180px; 
            padding-top: 0.1rem !important; 
            padding-bottom: 0rem !important;
            padding-left: 0.9rem !important;
            padding-right: 0.9rem !important;
            margin-top: 0 !important;
            position: relative; 
            z-index: 1; 
            background: transparent !important;
        }*/
        
        /* Target Streamlit's internal containers */
        div[data-testid="stAppViewContainer"],
        div[data-testid="stAppViewContainer"] > main,
        section.main,
        .main > div,
        div[data-testid="stVerticalBlock"],
        div[data-testid="stVerticalBlock"] > div {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        
        /* Override any remaining Streamlit spacing */
        .stApp > div:first-child,
        .stApp > div:first-child > div,
        .stApp > div:first-child > div > div {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        
        /* Force remove all top spacing from Streamlit elements */
        div[data-testid="stAppViewContainer"] > div,
        div[data-testid="stAppViewContainer"] > div > div,
        div[data-testid="stAppViewContainer"] > div > div > div {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        
        /* Target the first markdown element specifically */
        .main .block-container > div:first-child {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }

        /* Top Nav */
        .top-nav {
            position: fixed; top: 1.2rem; left: 50%; transform: translateX(-50%); z-index: 1000;
            display: flex; align-items: center; gap: 16px; justify-content: space-between;
            padding: 16px 20px; margin: 0;
            max-width: 1200px; width: calc(100% - 40px);
            background: linear-gradient(180deg, rgba(11,18,32,0.9), rgba(11,18,32,0.65));
            border: 1px solid rgba(96,165,250,0.15);
            border-top: 3px solid rgba(96, 165, 250, 0.75);
            border-radius: 14px;
            backdrop-filter: blur(8px);
            box-shadow: 0 -2px 8px rgba(96, 165, 250, 0.15), 0 4px 20px rgba(0, 0, 0, 0.2);
        }
        .brand { display: flex; align-items: center; gap: 12px; }
        .logo {
            height: 36px; width: 36px; border-radius: 9px;
            background: radial-gradient(120% 120% at 20% 20%, var(--primary-1), var(--primary-2) 60%, var(--primary-3));
            box-shadow: 0 6px 18px rgba(37,99,235,0.35), inset 0 0 10px rgba(255,255,255,0.12);
            display: flex; align-items: center; justify-content: center; font-weight: 800; color: white; letter-spacing: .5px;
        }
        .title-wrap { display: flex; flex-direction: column; gap: 2px; }
        .brand-title { font-weight: 800; letter-spacing: 0.3px; }
        .brand-subtitle { font-size: 12px; color: var(--muted); line-height: 1; }
        .brand-right { display: flex; align-items: center; }
        .brand-right img { 
            height: 28px; width: auto; display: block; opacity: .95; 
            filter: brightness(0) invert(1) contrast(1.1); /* make dark SVG visible on dark bg */
            mix-blend-mode: normal;
        }

        .nav-links { display: flex; gap: 14px; align-items: center; opacity: .95; }
        .nav-pill { color: var(--muted); padding: 6px 10px; border-radius: 8px; border: 1px solid transparent; }
        .nav-pill:hover { color: var(--text); border-color: rgba(96,165,250,0.25); background: rgba(37,99,235,0.08); }

        /* Hero */
        .hero {
            position: relative; z-index: 1; overflow: hidden; display: block;
            background:
                radial-gradient(60% 40% at 20% 20%, rgba(37,99,235,0.26), transparent 60%),
                radial-gradient(50% 35% at 80% 20%, rgba(96,165,250,0.26), transparent 60%),
                radial-gradient(70% 60% at 50% 90%, rgba(34,211,238,0.18), transparent 60%),
                linear-gradient(180deg, rgba(11,18,32,0.92), rgba(7,13,23,0.88));
            border: 1px solid rgba(147,197,253,0.45);
            border-radius: 18px;
            padding: 36px 28px;
            margin: 18px auto 14px auto; max-width: 1280px;
            box-shadow: 0 24px 48px rgba(2,6,23,0.55), 0 3px 10px rgba(37,99,235,0.18);
            min-height: 220px;
        }
        .hero, .hero * { color: var(--text) !important; }
        
        .hero .tagline {
            font-size: 1.35rem;
            font-weight: 600;
            background: linear-gradient(135deg, #60a5fa 0%, #22d3ee 50%, #93c5fd 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 12px 0 20px 0;
            letter-spacing: 0.3px;
            line-height: 1.4;
            text-align: center;
        }
        
        /* Animated floating light orbs */
        .hero::before,
        .hero::after {
            content: "";
            position: absolute;
            inset: -20% -10% -10% -10%;
            pointer-events: none;
            z-index: 0;
            filter: blur(22px) saturate(120%);
            opacity: 0.85;
            background:
                radial-gradient(180px 160px at 12% 25%, rgba(59,130,246,0.45), transparent 60%),
                radial-gradient(240px 200px at 85% 22%, rgba(96,165,250,0.38), transparent 65%),
                radial-gradient(260px 240px at 55% 92%, rgba(34,211,238,0.30), transparent 65%);
            transform: translate3d(0,0,0);
            animation: floatOrbs1 22s ease-in-out infinite alternate;
        }
        .hero::after {
            opacity: 0.65;
            filter: blur(28px) saturate(130%);
            background:
                radial-gradient(200px 180px at 20% 80%, rgba(59,130,246,0.35), transparent 62%),
                radial-gradient(300px 240px at 78% 60%, rgba(147,197,253,0.30), transparent 66%),
                radial-gradient(200px 180px at 48% 18%, rgba(14,165,233,0.30), transparent 62%);
            animation: floatOrbs2 26s ease-in-out infinite alternate;
        }
        .hero > * { position: relative; z-index: 1; }
        .hero h1 { margin: 0; font-size: 28px; font-weight: 900; letter-spacing: .2px; }
        .hero p { color: var(--muted); margin: 6px 0 0 0; max-width: 760px; }

        /* Feature grid */
        .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 20px; }
        @media (max-width: 1100px) { .grid { grid-template-columns: repeat(2, 1fr); } }
        @media (max-width: 640px) { .grid { grid-template-columns: 1fr; } }

        .card {
            position: relative;
            border: 1px solid rgba(96,165,250,0.22);
            /* Glassy transparent panel with soft blue tint */
            background: linear-gradient(180deg, rgba(20,30,55,0.28), rgba(20,30,55,0.16));
            -webkit-backdrop-filter: blur(8px) saturate(120%);
            backdrop-filter: blur(8px) saturate(120%);
            border-radius: 14px;
            padding: 18px 16px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 10px 26px rgba(2, 6, 23, 0.45), 0 6px 18px rgba(37,99,235,0.10);
            transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease, background-color 140ms ease;
            min-height: 120px;
        }
        .card:hover { 
            transform: translateY(-2px);
            border-color: rgba(147,197,253,0.55);
            background: linear-gradient(180deg, rgba(20,30,55,0.34), rgba(20,30,55,0.20));
            box-shadow: 0 18px 36px rgba(37,99,235,0.22), inset 0 1px 0 rgba(255,255,255,0.06);
        }
        .card .icon {
            height: 36px; width: 36px; border-radius: 10px;
            display: grid; place-items: center; color: white; font-size: 18px; font-weight: 800;
            background: radial-gradient(120% 120% at 20% 20%, var(--accent), var(--primary-1) 60%, var(--primary-2));
            box-shadow: 0 8px 22px rgba(34,211,238,0.25);
        }
        .card .title { margin-top: 10px; font-weight: 800; }
        .card .desc { margin-top: 4px; color: var(--muted); font-size: 14px; line-height: 1.45; }

        /* Metrics layout: 3 cols of metrics + 1 col map panel */
        .metrics-wrapper {
            display: grid; grid-template-columns: 3fr 1fr; gap: 12px;
            margin: 12px auto 0 auto; max-width: 1200px; padding: 0 16px;
        }
        .metrics-container {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
            margin: 0;
        }
        .map-panel {
            border: 1px solid rgba(96,165,250,0.22);
            background: linear-gradient(180deg, rgba(20,30,55,0.28), rgba(20,30,55,0.16));
            -webkit-backdrop-filter: blur(8px) saturate(120%);
            backdrop-filter: blur(8px) saturate(120%);
            border-radius: 14px 14px 0px 0px;
            padding: 10px 10px 8px 10px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
            margin-bottom: 0px;
        }
        .map-panel .map-title { font-weight: 800; margin-bottom: 0px; }
        .map-panel .coords { color: var(--muted); font-size: 13px; line-height: 1.3; margin-bottom: 0px; }
        
        /* Style for the map container to connect with the panel */
        div[data-testid="column"]:has(.map-panel) [data-testid="stDeckGlJsonChart"] {
            border: 1px solid rgba(96,165,250,0.22);
            border-top: none;
            border-radius: 0px 0px 14px 14px;
            overflow: hidden;
        }
        .map-placeholder {
            flex: 1; border: 1px dashed rgba(96,165,250,0.35); border-radius: 10px;
            display: grid; place-items: center; color: var(--muted);
            background: repeating-linear-gradient(45deg, rgba(59,130,246,0.06), rgba(59,130,246,0.06) 10px, rgba(59,130,246,0.03) 10px, rgba(59,130,246,0.03) 20px);
        }
        
        .metric-card {
            background: linear-gradient(135deg, rgba(17, 26, 44, 0.08), rgba(11, 18, 32, 0.12));
            border: 1px solid rgba(96, 165, 250, 0.15);
            border-left: 3px solid rgba(96, 165, 250, 0.75);
            border-radius: 12px;
            padding: 12px 30px 12px 8px;
            display: flex;
            flex-direction: row;
            align-items: center;
            text-align: center;
            gap: 6px;
            transition: all 0.2s ease;
            backdrop-filter: blur(12px) saturate(120%);
            box-shadow: -2px 0 8px rgba(96, 165, 250, 0.15), 0 4px 20px rgba(37, 99, 235, 0.08);
            height: 90px;
            width: 85%;
            justify-content: flex-start;
        }
        
        .metric-card:hover {
            transform: translateY(-4px);
            border-color: rgba(96, 165, 250, 0.4);
            border-left-color: rgba(96, 165, 250, 0.95);
            box-shadow: -4px 0 12px rgba(96, 165, 250, 0.25), 0 12px 40px rgba(37, 99, 235, 0.2);
        }
        
        .metric-icon {
            font-size: 20px;
            width: 24px;
            height: 24px;
            min-width: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: transparent;
            border-radius: 0px;
            border: none;
            margin-bottom: 0px;
            margin-right: 5px;
        }
        
        .metric-icon svg {
            width: 24px;
            height: 24px;
            fill: #60a5fa;
        }
        
        .metric-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            justify-content: center;
            gap: 2px;
        }
        
        .metric-label {
            font-size: 12px;
            color: var(--muted);
            margin-bottom: 2px;
            font-weight: 500;
            line-height: 1.2;
        }
        
        .metric-value {
            font-size: 18px;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 2px;
            line-height: 1.1;
        }
        
        .metric-change {
            font-size: 10px;
            font-weight: 600;
            padding: 2px 6px;
            border-radius: 4px;
            display: inline-block;
        }
        
        .metric-change.positive {
            background: rgba(96, 165, 250, 0.15);
            color: #60a5fa;
            border: 1px solid rgba(96, 165, 250, 0.25);
        }
        
        .metric-change.negative {
            background: rgba(239, 68, 68, 0.15);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.25);
        }
        
        /* Premium Driver Score Card Styling */
        .driver-score-premium {
            height: 120px !important;
            position: relative;
            overflow: visible;
        }
        
        .driver-score-premium::before {
            content: '';
            position: absolute;
            inset: -2px;
            border-radius: 12px;
            padding: 2px;
            background: linear-gradient(135deg, currentColor, transparent);
            -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
            animation: scoreGlow 3s ease-in-out infinite;
            pointer-events: none;
        }
        
        @keyframes scoreGlow {
            0%, 100% { opacity: 0.6; }
            50% { opacity: 1; }
        }
        
        .score-grade-badge {
            position: absolute;
            top: 10px;
            right: 15px;
            font-size: 16px;
            font-weight: 800;
            padding: 4px 10px;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.1);
            border: 2px solid rgba(255, 255, 255, 0.3);
            backdrop-filter: blur(4px);
        }
        
        .score-trend-indicator {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            font-size: 11px;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 6px;
            margin-left: 8px;
        }
        
        .trend-up {
            background: rgba(34, 197, 94, 0.15);
            color: #22c55e;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }
        
        .trend-down {
            background: rgba(239, 68, 68, 0.15);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        
        .trend-neutral {
            background: rgba(251, 191, 36, 0.15);
            color: #fbbf24;
            border: 1px solid rgba(251, 191, 36, 0.3);
        }
        
        @media (max-width: 1200px) {
            .metrics-container {
                grid-template-columns: repeat(2, 1fr);
                gap: 18px;
                padding: 0 16px;
            }
        }
        @media (max-width: 768px) {
            .metrics-container {
                grid-template-columns: 1fr;
                gap: 18px;
                padding: 0 16px;
            }
        }

        /* Footer */
        .footer { margin: 24px auto 12px auto; max-width: 1280px; color: var(--muted); font-size: 12px; text-align: center; opacity: .85; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_nav() -> None:
    # Build Nissan logo data URI if available
    nissan_logo_src = ""
    try:
        logo_path = Path(__file__).parent / "images" / "nissan_logo.svg"
        with open(logo_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
            nissan_logo_src = f"data:image/svg+xml;base64,{encoded}"
    except Exception:
        nissan_logo_src = ""

    st.markdown(
        f"""
        <div class="page-bg">
            <div class="drift-left"></div>
            <div class="drift-right"></div>
            <div class="float-extra-1"></div>
            <div class="float-extra-2"></div>
            <div class="float-extra-3"></div>
        </div>
        <div class="top-nav">
            <div class="brand">
                <div class="logo">UBI</div>
                <div class="title-wrap">
                    <div class="brand-title">Usage-Based Insurance Dashboard</div>
                    <div class="brand-subtitle">Turning driving behavior into smarter coverage</div>
                </div>
            </div>
            <div class="brand-right">{('<img src="' + nissan_logo_src + '" alt="Nissan" />') if nissan_logo_src else ''}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_showcase() -> None:
    # Controls row: VIN and Period
    # Spacer to separate from fixed navbar
    st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)
    vin_options = [
        "1N4AA5AP7DC123456", "3FA6P0G72GR789012", "5YJSA1E26HF345678",
        "1HGCM82633A004352", "2T3BF4DV4BW112233", "JTDKN3DU0A0123456",
        "SALVP2BG1FH123987", "WBA3A5C54CF678901", "1FTFW1EF1EFA12345",
        "4T1BF1FK5FU765432"
    ]
    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        l1, i1, spacing = st.columns([0.08, 0.60, 0.1,])
        with l1:
            st.markdown("<div style='display:flex;align-items:center;height:36px;'>VIN</div>", unsafe_allow_html=True)
        with i1:
            st.selectbox("", vin_options, index=0, key="vin_select", label_visibility="collapsed")
    with c2:
        default_end = date.today()
        default_start = default_end - timedelta(days=30)
        l2, i2, spacing = st.columns([0.18, 0.72, 0.1,])
        with l2:
            st.markdown("<div style='display:flex;align-items:center;height:36px;'>Period</div>", unsafe_allow_html=True)
        with i2:
            # Restrict date selection: min = 1 year ago, max = today
            min_date = date.today() - timedelta(days=365)
            st.date_input("", (default_start, default_end), min_value=min_date, max_value=date.today(), key="period_select", label_visibility="collapsed")

    # Compute dynamic metrics and NA coordinates based on VIN and period
    selected_vin = st.session_state.get("vin_select", vin_options[0])
    selected_period = st.session_state.get("period_select", (default_start, default_end))
    
    # Validate that both start and end dates are selected and not in the future
    if not isinstance(selected_period, tuple) or len(selected_period) != 2:
        st.warning("⚠️ Please select both start and end dates for the period.")
        st.stop()
    
    start_date, end_date = selected_period
    today = date.today()
    
    # Check if dates are in the future
    if start_date > today or end_date > today:
        st.error("❌ Future dates are not allowed. Please select dates up to today.")
        st.stop()
    
    if start_date > end_date:
        st.error("❌ Start date must be before or equal to end date.")
        st.stop()
    
    seed_val = abs(hash((selected_vin, str(selected_period)))) % (10**9)
    rng = random.Random(seed_val)

    # Calculate period length for scaling metrics appropriately
    days_in_period = (end_date - start_date).days + 1  # +1 to include both start and end dates
    
    # Scale metrics based on period length (baseline: 30 days)
    period_scale = days_in_period / 30.0
    
    # We'll calculate distance_travelled and drive_time after generating the routes
    
    # === REALISTIC METRICS FOR 30-DAY BASELINE ===
    # All values based on real-world driving statistics
    
    # Idle Time: Realistic range for 30 days (engine running while stopped)
    # Average driver: 20-60 hours/month of idling (traffic, warming up, waiting)
    base_idle_time = rng.uniform(15.0, 50.0)
    idle_time = round(base_idle_time * period_scale, 1)
    
    # Hard Braking Events: 0-1 per day on average (0-30 per month)
    # Good drivers: 5-15/month, Average: 15-30/month
    base_hard_braking = rng.randint(5, 30)
    hard_braking_events = max(1, int(base_hard_braking * period_scale))
    
    # Hard Acceleration Events: Similar to braking
    # Good drivers: 5-20/month, Average: 20-40/month
    base_hard_accel = rng.randint(5, 35)
    hard_acceleration_events = max(1, int(base_hard_accel * period_scale))
    
    # Speeding Events: Most common safety event
    # Good drivers: 10-30/month, Average: 30-60/month
    base_speeding = rng.randint(10, 55)
    speeding_events = max(1, int(base_speeding * period_scale))
    
    # Idle Events: Number of times vehicle was left idling (not total time)
    # Typical: 30-100 idle events per month (1-3 per day)
    base_idle_events = rng.randint(30, 90)
    idle_events = max(1, int(base_idle_events * period_scale))
    
    # Driver Experience: Years of driving experience - doesn't scale with period
    # Realistic range: 2-40 years (new drivers to very experienced)
    # Most drivers: 5-25 years
    driver_experience = round(rng.uniform(5.0, 30.0), 1)
    
    # Driver Accident History: Very rare for most drivers
    # 85% of drivers: 0 accidents per month
    # 10% of drivers: 1 accident per month
    # 5% of drivers: 2+ accidents per month
    accident_chance = rng.random()
    if accident_chance < 0.85:
        base_accidents = 0  # 85% chance: no accidents
    elif accident_chance < 0.95:
        base_accidents = 1  # 10% chance: 1 accident
    else:
        base_accidents = 2  # 5% chance: 2 accidents
    accident_history = int(base_accidents * period_scale)
    
    # Night Driving Hours: 10pm-6am driving (8-hour window)
    # Average: 15-25% of total drive time occurs at night
    # For 30-40h drive time, expect 4-10h night driving
    base_night_driving = rng.uniform(4.0, 12.0)
    night_driving_hours = round(base_night_driving * period_scale, 1)
    
    api_calls_m = rng.uniform(0.8, 2.4)
    model_updates = rng.randint(1, 6)

    # Real landmarks and points of interest in North America
    REAL_LANDMARKS = {
        "seattle": [
            {"name": "Space Needle", "lat": 47.6205, "lng": -122.3493, "type": "landmark"},
            {"name": "Pike Place Market", "lat": 47.6097, "lng": -122.3421, "type": "market"},
            {"name": "Seattle-Tacoma Airport", "lat": 47.4502, "lng": -122.3088, "type": "airport"},
            {"name": "University of Washington", "lat": 47.6553, "lng": -122.3035, "type": "university"},
            {"name": "Amazon HQ", "lat": 47.6219, "lng": -122.3390, "type": "office"},
            {"name": "Alki Beach", "lat": 47.5806, "lng": -122.4079, "type": "beach"},
        ],
        "sanfrancisco": [
            {"name": "Golden Gate Bridge", "lat": 37.8199, "lng": -122.4783, "type": "landmark"},
            {"name": "Fisherman's Wharf", "lat": 37.8080, "lng": -122.4177, "type": "attraction"},
            {"name": "San Francisco Airport", "lat": 37.6213, "lng": -122.3790, "type": "airport"},
            {"name": "Oracle Park", "lat": 37.7786, "lng": -122.3893, "type": "stadium"},
            {"name": "Union Square", "lat": 37.7880, "lng": -122.4075, "type": "plaza"},
            {"name": "Alcatraz Island", "lat": 37.8270, "lng": -122.4230, "type": "landmark"},
        ],
        "losangeles": [
            {"name": "Hollywood Sign", "lat": 34.1341, "lng": -118.3215, "type": "landmark"},
            {"name": "LAX Airport", "lat": 33.9416, "lng": -118.4085, "type": "airport"},
            {"name": "Santa Monica Pier", "lat": 34.0095, "lng": -118.4974, "type": "pier"},
            {"name": "Dodger Stadium", "lat": 34.0739, "lng": -118.2400, "type": "stadium"},
            {"name": "Getty Center", "lat": 34.0780, "lng": -118.4741, "type": "museum"},
            {"name": "Venice Beach", "lat": 33.9850, "lng": -118.4695, "type": "beach"},
        ],
        "chicago": [
            {"name": "Willis Tower", "lat": 41.8789, "lng": -87.6359, "type": "landmark"},
            {"name": "Navy Pier", "lat": 41.8917, "lng": -87.6086, "type": "pier"},
            {"name": "O'Hare Airport", "lat": 41.9742, "lng": -87.9073, "type": "airport"},
            {"name": "Millennium Park", "lat": 41.8826, "lng": -87.6226, "type": "park"},
            {"name": "Wrigley Field", "lat": 41.9484, "lng": -87.6553, "type": "stadium"},
            {"name": "Lincoln Park Zoo", "lat": 41.9212, "lng": -87.6334, "type": "zoo"},
        ],
        "newyork": [
            {"name": "Statue of Liberty", "lat": 40.6892, "lng": -74.0445, "type": "landmark"},
            {"name": "Times Square", "lat": 40.7580, "lng": -73.9855, "type": "plaza"},
            {"name": "JFK Airport", "lat": 40.6413, "lng": -73.7781, "type": "airport"},
            {"name": "Central Park", "lat": 40.7829, "lng": -73.9654, "type": "park"},
            {"name": "Empire State Building", "lat": 40.7484, "lng": -73.9857, "type": "landmark"},
            {"name": "Brooklyn Bridge", "lat": 40.7061, "lng": -73.9969, "type": "bridge"},
        ],
        "miami": [
            {"name": "South Beach", "lat": 25.7907, "lng": -80.1300, "type": "beach"},
            {"name": "Miami Airport", "lat": 25.7959, "lng": -80.2870, "type": "airport"},
            {"name": "Bayside Marketplace", "lat": 25.7789, "lng": -80.1862, "type": "market"},
            {"name": "Vizcaya Museum", "lat": 25.7443, "lng": -80.2106, "type": "museum"},
            {"name": "Wynwood Walls", "lat": 25.8010, "lng": -80.1994, "type": "art"},
            {"name": "Port of Miami", "lat": 25.7743, "lng": -80.1657, "type": "port"},
        ],
        "denver": [
            {"name": "Red Rocks Park", "lat": 39.6654, "lng": -105.2057, "type": "park"},
            {"name": "Denver Airport", "lat": 39.8561, "lng": -104.6737, "type": "airport"},
            {"name": "Coors Field", "lat": 39.7559, "lng": -104.9942, "type": "stadium"},
            {"name": "Denver Art Museum", "lat": 39.7372, "lng": -104.9894, "type": "museum"},
            {"name": "16th Street Mall", "lat": 39.7470, "lng": -104.9913, "type": "mall"},
            {"name": "Union Station", "lat": 39.7539, "lng": -105.0000, "type": "station"},
        ],
        "boston": [
            {"name": "Fenway Park", "lat": 42.3467, "lng": -71.0972, "type": "stadium"},
            {"name": "Boston Common", "lat": 42.3551, "lng": -71.0656, "type": "park"},
            {"name": "Logan Airport", "lat": 42.3656, "lng": -71.0096, "type": "airport"},
            {"name": "Harvard University", "lat": 42.3770, "lng": -71.1167, "type": "university"},
            {"name": "Quincy Market", "lat": 42.3601, "lng": -71.0547, "type": "market"},
            {"name": "New England Aquarium", "lat": 42.3591, "lng": -71.0492, "type": "aquarium"},
        ],
    }
    
    # Select city based on VIN hash
    def select_city_for_vin(vin: str) -> str:
        """Select a city based on VIN hash."""
        city_names = list(REAL_LANDMARKS.keys())
        city_idx = abs(hash(vin)) % len(city_names)
        return city_names[city_idx]
    
    # Generate REAL street-following route using OSRM (Open Source Routing Machine)
    @st.cache_data(ttl=3600)  # Cache routes for 1 hour to avoid repeated API calls
    def get_real_driving_route(start_place: dict, end_place: dict) -> tuple:
        """
        Get actual driving route following real streets using OSRM API.
        Returns: (route_coords, distance_km, duration_minutes)
        """
        start_lng, start_lat = start_place["lng"], start_place["lat"]
        end_lng, end_lat = end_place["lng"], end_place["lat"]
        
        # Open Source Routing Machine (OSRM) a public API endpoint for driving routes
        url = f"http://router.project-osrm.org/route/v1/driving/{start_lng},{start_lat};{end_lng},{end_lat}"
        params = {
            "overview": "full",  # Get full route geometry
            "geometries": "geojson",  # Return as GeoJSON coordinates
            "steps": "true"  # Include turn-by-turn instructions
        }
        
        try:
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("code") == "Ok" and data.get("routes"):
                    route = data["routes"][0]
                    
                    # Extract route geometry (already in [lng, lat] format)
                    coordinates = route["geometry"]["coordinates"]
                    
                    # Extract distance and duration
                    distance_km = round(route["distance"] / 1000, 1)  # Convert meters to km
                    duration_min = round(route["duration"] / 60, 0)  # Convert seconds to minutes
                    
                    return coordinates, distance_km, duration_min
            
            # Fallback: if API fails, create a simple straight line
            return [[start_lng, start_lat], [end_lng, end_lat]], 0, 0
            
        except Exception as e:
            # Fallback on error
            st.warning(f"Route API temporarily unavailable, using simplified route visualization.")
            return [[start_lng, start_lat], [end_lng, end_lat]], 0, 0
    
    # ===== GENERATE ROUTES FIRST TO CALCULATE TOTAL DISTANCE =====
    # Select real city and landmarks for this VIN
    selected_city = select_city_for_vin(selected_vin)
    city_landmarks = REAL_LANDMARKS[selected_city]
    
    # Select 3 pairs of landmarks for routes (deterministic based on period)
    period_seed = abs(hash(str(selected_period))) % (10**6)
    route_rng = random.Random(seed_val + period_seed)
    
    # Shuffle and pick pairs
    available_landmarks = city_landmarks.copy()
    route_rng.shuffle(available_landmarks)
    
    # Ensure we have at least 6 landmarks
    while len(available_landmarks) < 6:
        available_landmarks.extend(city_landmarks)
    
    # Create 3 route pairs
    route1_start = available_landmarks[0]
    route1_end = available_landmarks[1]
    route2_start = available_landmarks[2]
    route2_end = available_landmarks[3]
    route3_start = available_landmarks[4]
    route3_end = available_landmarks[5]
    
    # Generate REAL street-following routes using OSRM API
    with st.spinner('🗺️ Loading real street routes...'):
        route1_path, route1_km, route1_duration = get_real_driving_route(route1_start, route1_end)
        time.sleep(0.1)  # Small delay to respect API rate limits
        route2_path, route2_km, route2_duration = get_real_driving_route(route2_start, route2_end)
        time.sleep(0.1)
        route3_path, route3_km, route3_duration = get_real_driving_route(route3_start, route3_end)
    
    # Convert distances to miles (1 km = 0.621371 miles)
    KM_TO_MILES = 0.621371
    route1_miles = round(route1_km * KM_TO_MILES, 1)
    route2_miles = round(route2_km * KM_TO_MILES, 1)
    route3_miles = round(route3_km * KM_TO_MILES, 1)
    
    # Calculate realistic total distance travelled for the period
    # The 3 routes shown are just SAMPLE trips, not the only trips
    # Average US driver: ~13,500 miles/year = ~1,125 miles/month = ~37 miles/day
    # We'll generate realistic monthly mileage based on period length
    base_daily_miles = rng.uniform(25.0, 45.0)  # 25-45 miles per day (realistic range)
    distance_travelled = round(base_daily_miles * days_in_period, 1)
    
    # Calculate realistic total drive time for the period
    # The 3 routes shown are just SAMPLE trips
    # Average speed considering city/highway mix: 30-35 mph
    # Drive time = Distance / Speed
    avg_speed = rng.uniform(28.0, 35.0)  # Realistic average speed (mph)
    drive_time = round(distance_travelled / avg_speed, 1)  # Total hours for the period
    
    # Calculate Driver Score (0-100) - comprehensive assessment based on ALL metrics
    # This is the FINAL score representing overall driver quality
    driver_score = 100.0
    
    # Normalize events to per-30-day basis for fair scoring across different periods
    normalized_scale = 30.0 / days_in_period
    norm_braking = hard_braking_events * normalized_scale
    norm_accel = hard_acceleration_events * normalized_scale
    norm_speeding = speeding_events * normalized_scale
    norm_idle_events = idle_events * normalized_scale
    norm_accidents = accident_history * normalized_scale
    norm_idle_time = idle_time * normalized_scale
    norm_night_driving = night_driving_hours * normalized_scale
    
    # Deduct points based on safety/behavior metrics (weighted by severity/importance)
    # Adjusted weights for realistic event frequencies (events are now more common)
    driver_score -= (norm_braking * 0.25)       # Hard braking: -0.25 points each
    driver_score -= (norm_accel * 0.20)         # Hard acceleration: -0.20 points each
    driver_score -= (norm_speeding * 0.35)      # Speeding: -0.35 points each (highest event penalty)
    driver_score -= (norm_idle_events * 0.05)   # Idle events: -0.05 points each
    driver_score -= (norm_idle_time * 0.15)     # Idle time: -0.15 points per hour
    driver_score -= (norm_night_driving * 0.50) # Night driving: -0.50 points per hour (HIGH risk)
    driver_score -= (norm_accidents * 20.0)     # Accidents: -20 points each (SEVERE penalty)
    
    # Impact of Driver Experience (years of driving) on risk
    # Based on insurance actuarial data - follows U-shaped risk curve
    # Young/inexperienced drivers (< 5 years) = HIGH risk → penalty
    # Prime experienced drivers (10-25 years) = LOW risk → bonus
    # Very senior drivers (30+ years) = MODERATE risk → smaller bonus (slower reflexes)
    
    if driver_experience < 3:
        # Very new drivers: 0-3 years (HIGH risk - inexperienced)
        experience_adjustment = -8.0
    elif driver_experience < 5:
        # New drivers: 3-5 years (MODERATE risk - still learning)
        experience_adjustment = -3.0
    elif driver_experience < 10:
        # Developing experience: 5-10 years (gaining skill)
        experience_adjustment = (driver_experience - 5.0) * 0.8  # 0 to +4 pts
    elif driver_experience < 25:
        # Prime experience: 10-25 years (LOWEST risk - peak performance)
        experience_adjustment = 4.0 + ((driver_experience - 10.0) / 15.0) * 6.0  # +4 to +10 pts
    else:
        # Senior drivers: 25+ years (MODERATE risk - experience vs age effects)
        experience_adjustment = 8.0  # Cap at +8 (still good, but not peak)
    
    driver_score += experience_adjustment
    
    # Ensure score stays within 0-100 range
    driver_score = max(0, min(100, round(driver_score, 1)))
    
    # Determine score tier for dynamic styling
    if driver_score >= 90:
        score_tier = "excellent"
        score_grade = "A+"
        score_color_start = "rgba(34, 197, 94, 0.20)"  # Green
        score_color_end = "rgba(22, 163, 74, 0.12)"
        score_border = "rgba(34, 197, 94, 0.5)"
        score_glow = "0 0 25px rgba(34, 197, 94, 0.4)"
        score_label = "⭐ Excellent Driver"
        score_trend = f"+{rng.uniform(2.0, 5.0):.1f}"
    elif driver_score >= 75:
        score_tier = "good"
        score_grade = "B+"
        score_color_start = "rgba(59, 130, 246, 0.20)"  # Blue
        score_color_end = "rgba(37, 99, 235, 0.12)"
        score_border = "rgba(96, 165, 250, 0.5)"
        score_glow = "0 0 25px rgba(96, 165, 250, 0.4)"
        score_label = "✓ Good Driver"
        score_trend = f"+{rng.uniform(1.0, 3.0):.1f}"
    elif driver_score >= 60:
        score_tier = "average"
        score_grade = "C"
        score_color_start = "rgba(251, 191, 36, 0.18)"  # Amber
        score_color_end = "rgba(245, 158, 11, 0.10)"
        score_border = "rgba(251, 191, 36, 0.45)"
        score_glow = "0 0 25px rgba(251, 191, 36, 0.35)"
        score_label = "△ Average Driver"
        score_trend = f"{rng.uniform(-1.5, 1.5):.1f}"
    else:
        score_tier = "poor"
        score_grade = "D"
        score_color_start = "rgba(239, 68, 68, 0.18)"  # Red
        score_color_end = "rgba(220, 38, 38, 0.10)"
        score_border = "rgba(239, 68, 68, 0.45)"
        score_glow = "0 0 25px rgba(239, 68, 68, 0.35)"
        score_label = "⚠ Needs Improvement"
        score_trend = f"-{rng.uniform(2.0, 5.0):.1f}"

    # Create a 2-column layout: metrics on left (5fr), map on right (2fr)
    col_metrics, col_map = st.columns([5, 2])
    
    with col_metrics:
        st.markdown(
            f"""
            <div class="metrics-container">
                <div class="metric-card driver-score-premium" style="grid-column: span 2; background: linear-gradient(135deg, {score_color_start}, {score_color_end}); border: 2px solid {score_border}; box-shadow: {score_glow}, -2px 0 8px rgba(96, 165, 250, 0.15), 0 4px 20px rgba(37, 99, 235, 0.08);">
                    <div class="score-grade-badge">{score_grade}</div>
                    <div class="metric-icon" style="width: 32px; height: 32px; min-width: 32px;">
                        <svg viewBox="0 0 24 24" style="width: 32px; height: 32px;">
                            <path d="M12,15L16.33,17.5L15.5,12.7L19,9.24L14.19,8.63L12,4.19L9.81,8.63L5,9.24L8.5,12.7L7.67,17.5L12,15M22,9.24L14.81,8.63L12,2L9.19,8.63L2,9.24L7.45,13.97L5.82,21L12,17.27L18.18,21L16.54,13.97L22,9.24Z"/>
                        </svg>
                    </div>
                    <div class="metric-content">
                        <div class="metric-label" style="font-size: 13px; font-weight: 600; letter-spacing: 0.3px;">🏆 DRIVER SCORE</div>
                        <div class="metric-value" style="font-size: 28px; font-weight: 900; letter-spacing: -0.5px; margin: 4px 0;">{driver_score:.1f}<span style="font-size: 18px; opacity: 0.7;">/100</span></div>
                        <div style="display: flex; align-items: center; justify-content: center; gap: 6px;">
                            <div class="metric-change {'positive' if driver_score >= 75 else 'negative' if driver_score < 60 else 'positive'}" style="font-size: 11px;">{score_label}</div>
                            <div class="score-trend-indicator {'trend-up' if float(score_trend) > 0 else 'trend-down' if float(score_trend) < 0 else 'trend-neutral'}">
                                {'↗' if float(score_trend) > 0 else '↘' if float(score_trend) < 0 else '→'} {score_trend}pts
                    </div>
                </div>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">
                        <svg viewBox="0 0 24 24">
                            <path d="M12,2L4.5,20.29L5.21,21L12,18L18.79,21L19.5,20.29L12,2M6.12,17L12,5.5L17.88,17L12,14.5L6.12,17M18,17H19V18H18V17Z"/>
                        </svg>
                    </div>
                    <div class="metric-content">
                        <div class="metric-label">Distance Travelled</div>
                        <div class="metric-value">{distance_travelled:.1f} mi</div>
                        <div class="metric-change positive">+{rng.uniform(3.0,8.0):.1f}%</div>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">
                        <svg viewBox="0 0 24 24">
                            <path d="M12,20A7,7 0 0,1 5,13A7,7 0 0,1 12,6A7,7 0 0,1 19,13A7,7 0 0,1 12,20M19.03,7.39L20.45,5.97C20,5.46 19.55,5 19.04,4.56L17.62,6C16.07,4.74 14.12,4 12,4A9,9 0 0,0 3,13A9,9 0 0,0 12,22C17,22 21,17.97 21,13C21,10.88 20.26,8.93 19.03,7.39M11,14H13V8H11M15,1H9V3H15V1Z"/>
                        </svg>
                    </div>
                    <div class="metric-content">
                        <div class="metric-label">Idle Time</div>
                        <div class="metric-value">{idle_time:.1f}h</div>
                        <div class="metric-change negative">-{rng.uniform(5.0,15.0):.1f}%</div>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">
                        <svg viewBox="0 0 24 24">
                            <path d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M16.2,16.2L11,13V7H12.5V12.2L17,14.9L16.2,16.2Z"/>
                        </svg>
                    </div>
                    <div class="metric-content">
                        <div class="metric-label">Drive Time</div>
                        <div class="metric-value">{drive_time:.1f}h</div>
                        <div class="metric-change positive">+{rng.uniform(2.0,8.0):.1f}%</div>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">
                        <svg viewBox="0 0 24 24">
                            <path d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,4A8,8 0 0,1 20,12A8,8 0 0,1 12,20A8,8 0 0,1 4,12A8,8 0 0,1 12,4M11,7V13H13V7H11M11,15V17H13V15H11Z"/>
                        </svg>
                    </div>
                    <div class="metric-content">
                        <div class="metric-label">Hard Braking Events</div>
                        <div class="metric-value">{hard_braking_events}</div>
                        <div class="metric-change negative">-{rng.uniform(10.0,25.0):.1f}%</div>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">
                        <svg viewBox="0 0 24 24">
                            <path d="M12,2C6.5,2 2,6.5 2,12C2,17.5 6.5,22 12,22C17.5,22 22,17.5 22,12C22,6.5 17.5,2 12,2M12,4C16.4,4 20,7.6 20,12C20,16.4 16.4,20 12,20C7.6,20 4,16.4 4,12C4,7.6 7.6,4 12,4M13,7L11,7L11,13L13,13L13,7M15.5,8.5L14,10L17,13L18.5,11.5L15.5,8.5Z"/>
                        </svg>
                    </div>
                    <div class="metric-content">
                        <div class="metric-label">Hard Acceleration</div>
                        <div class="metric-value">{hard_acceleration_events}</div>
                        <div class="metric-change negative">-{rng.uniform(8.0,20.0):.1f}%</div>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">
                        <svg viewBox="0 0 24 24">
                            <path d="M18.92,6.01C18.72,5.42 18.16,5 17.5,5H6.5C5.84,5 5.29,5.42 5.08,6.01L3,12V20C3,20.55 3.45,21 4,21H5C5.55,21 6,20.55 6,20V19H18V20C18,20.55 18.45,21 19,21H20C20.55,21 21,20.55 21,20V12L18.92,6.01M6.5,16C5.67,16 5,15.33 5,14.5C5,13.67 5.67,13 6.5,13C7.33,13 8,13.67 8,14.5C8,15.33 7.33,16 6.5,16M17.5,16C16.67,16 16,15.33 16,14.5C16,13.67 16.67,13 17.5,13C18.33,13 19,13.67 19,14.5C19,15.33 18.33,16 17.5,16M5,11L6.5,6.5H17.5L19,11H5M10,8V10H14V8H10Z"/>
                        </svg>
                    </div>
                    <div class="metric-content">
                        <div class="metric-label">Speeding Events</div>
                        <div class="metric-value">{speeding_events}</div>
                        <div class="metric-change negative">-{rng.uniform(5.0,18.0):.1f}%</div>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">
                        <svg viewBox="0 0 24 24">
                            <path d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M7.07,18.28C7.5,17.38 10.12,16.5 12,16.5C13.88,16.5 16.5,17.38 16.93,18.28C15.57,19.36 13.86,20 12,20C10.14,20 8.43,19.36 7.07,18.28M18.36,16.83C16.93,15.09 13.46,14.5 12,14.5C10.54,14.5 7.07,15.09 5.64,16.83C4.62,15.5 4,13.82 4,12C4,7.59 7.59,4 12,4C16.41,4 20,7.59 20,12C20,13.82 19.38,15.5 18.36,16.83M12,6C10.06,6 8.5,7.56 8.5,9.5C8.5,11.44 10.06,13 12,13C13.94,13 15.5,11.44 15.5,9.5C15.5,7.56 13.94,6 12,6M12,11A1.5,1.5 0 0,1 10.5,9.5A1.5,1.5 0 0,1 12,8A1.5,1.5 0 0,1 13.5,9.5A1.5,1.5 0 0,1 12,11Z"/>
                        </svg>
                    </div>
                    <div class="metric-content">
                        <div class="metric-label">Idle Events</div>
                        <div class="metric-value">{idle_events}</div>
                        <div class="metric-change negative">-{rng.uniform(12.0,22.0):.1f}%</div>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">
                        <svg viewBox="0 0 24 24">
                            <path d="M12,1L3,5V11C3,16.55 6.84,21.74 12,23C17.16,21.74 21,16.55 21,11V5M12,5A3,3 0 0,1 15,8A3,3 0 0,1 12,11A3,3 0 0,1 9,8A3,3 0 0,1 12,5M17.13,17C15.92,18.85 14.11,20.24 12,20.92C9.89,20.24 8.08,18.85 6.87,17C6.53,16.5 6.24,16 6,15.47C6,13.82 8.71,12.47 12,12.47C15.29,12.47 18,13.79 18,15.47C17.76,16 17.47,16.5 17.13,17Z"/>
                        </svg>
                    </div>
                    <div class="metric-content">
                        <div class="metric-label">Driver Experience</div>
                        <div class="metric-value">{driver_experience:.1f}<span style="font-size: 14px; opacity: 0.7;"> yrs</span></div>
                        <div class="metric-change positive">Experienced</div>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">
                        <svg viewBox="0 0 24 24">
                            <path d="M12,2L1,21H23M12,6L19.53,19H4.47M11,10V14H13V10M11,16V18H13V16"/>
                        </svg>
                    </div>
                    <div class="metric-content">
                        <div class="metric-label">Accident History</div>
                        <div class="metric-value">{accident_history}</div>
                        <div class="metric-change {'positive' if accident_history == 0 else 'negative'}">{'0 accidents' if accident_history == 0 else f'-{rng.uniform(15,35):.1f}%'}</div>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">
                        <svg viewBox="0 0 24 24">
                            <path d="M17.75,4.09L15.22,6.03L16.13,9.09L13.5,7.28L10.87,9.09L11.78,6.03L9.25,4.09L12.44,4L13.5,1L14.56,4L17.75,4.09M21.25,11L19.61,12.25L20.2,14.23L18.5,13.06L16.8,14.23L17.39,12.25L15.75,11L17.81,10.95L18.5,9L19.19,10.95L21.25,11M18.97,15.95C19.8,15.87 20.69,17.05 20.16,17.8C19.84,18.25 19.5,18.67 19.08,19.07C15.17,23 8.84,23 4.94,19.07C1.03,15.17 1.03,8.83 4.94,4.93C5.34,4.53 5.76,4.17 6.21,3.85C6.96,3.32 8.14,4.21 8.06,5.04C7.79,7.9 8.75,10.87 10.95,13.06C13.14,15.26 16.1,16.22 18.97,15.95M17.33,17.97C14.5,17.81 11.7,16.64 9.53,14.5C7.36,12.31 6.2,9.5 6.04,6.68C3.23,9.82 3.34,14.64 6.35,17.66C9.37,20.67 14.19,20.78 17.33,17.97Z"/>
                        </svg>
                    </div>
                    <div class="metric-content">
                        <div class="metric-label">Night Driving</div>
                        <div class="metric-value">{night_driving_hours:.1f}h</div>
                        <div class="metric-change negative">-{rng.uniform(10.0,25.0):.1f}%</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col_map:
        # Format duration for display
        def format_duration(minutes):
            if minutes < 60:
                return f"{int(minutes)}min"
            else:
                hours = int(minutes // 60)
                mins = int(minutes % 60)
                return f"{hours}h {mins}min" if mins > 0 else f"{hours}h"
        
        route1_time = format_duration(route1_duration)
        route2_time = format_duration(route2_duration)
        route3_time = format_duration(route3_duration)
        
        st.markdown(
            f"""
            <div class="map-panel" style="margin-bottom: 0px; padding-bottom: 4px;">
                <div class="coords" style="font-size: 12px; margin: 0px; padding: 0px; line-height: 1.3;">
                    <span style="color: #4287F5;">●</span> <b>{route1_start['name']}</b> → {route1_end['name']} <span style="opacity: 0.8;">({route1_miles}mi • {route1_time})</span><br/>
                    <span style="color: #22C55E;">●</span> <b>{route2_start['name']}</b> → {route2_end['name']} <span style="opacity: 0.8;">({route2_miles}mi • {route2_time})</span><br/>
                    <span style="color: #F59F42;">●</span> <b>{route3_start['name']}</b> → {route3_end['name']} <span style="opacity: 0.8;">({route3_miles}mi • {route3_time})</span>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Apply CSS to connect map panel and pydeck chart
        st.markdown(
            """
            <style>
            /* Remove gap between map-panel and pydeck chart */
            div[data-testid="column"]:has(.map-panel) > div[data-testid="stVerticalBlock"] {
                gap: 0rem !important;
            }
            
            div[data-testid="column"]:has(.map-panel) > div[data-testid="stVerticalBlock"] > div {
                margin: 0px !important;
                padding: 0px !important;
            }
            
            /* Make pydeck chart connect seamlessly with map-panel */
            div[data-testid="column"]:has(.map-panel) [data-testid="stDeckGlJsonChart"] {
                margin-top: 0px !important;
                padding-top: 0px !important;
            }
            
            /* Ensure the container of pydeck has no spacing */
            div[data-testid="column"]:has(.map-panel) div:has([data-testid="stDeckGlJsonChart"]) {
                margin-top: 0px !important;
                padding-top: 0px !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        
        # Render the map with route
        # Prepare route data for PathLayer with real street paths
        routes_data = pd.DataFrame([
            {
                "path": route1_path,
                "name": f"Route A: {route1_start['name']} → {route1_end['name']}",
                "color": [66, 135, 245, 200],  # Blue
                "width": 9,
                "info": f"🔵 Route A\n{route1_start['name']} → {route1_end['name']}\n{route1_miles} mi • {route1_time}"
            },
            {
                "path": route2_path,
                "name": f"Route B: {route2_start['name']} → {route2_end['name']}", 
                "color": [34, 197, 94, 200],  # Green
                "width": 9,
                "info": f"🟢 Route B\n{route2_start['name']} → {route2_end['name']}\n{route2_miles} mi • {route2_time}"
            },
            {
                "path": route3_path,
                "name": f"Route C: {route3_start['name']} → {route3_end['name']}",
                "color": [245, 159, 66, 200],  # Orange
                "width": 9,
                "info": f"🟠 Route C\n{route3_start['name']} → {route3_end['name']}\n{route3_miles} mi • {route3_time}"
            }
        ])
        
        # Create path layer for routes
        path_layer = pdk.Layer(
            "PathLayer",
            data=routes_data,
            get_path="path",
            get_width="width",
            get_color="color",
            width_min_pixels=3,
            width_scale=1,
            rounded=True,
            pickable=True,
            auto_highlight=True,
        )
        
        # Prepare marker data for start/end points with real landmark names
        markers_data = pd.DataFrame([
            # Route 1 markers
            {"lon": route1_start["lng"], "lat": route1_start["lat"], 
             "type": "start", "route": "Route A", "color": [66, 135, 245, 255],
             "name": route1_start["name"], 
             "info": f"🔵 START\n{route1_start['name']}\n{route1_start['type'].title()}\n\n→ {route1_end['name']}\n{route1_miles} mi • {route1_time}"},
            {"lon": route1_end["lng"], "lat": route1_end["lat"], 
             "type": "end", "route": "Route A", "color": [66, 135, 245, 255],
             "name": route1_end["name"],
             "info": f"🔵 END\n{route1_end['name']}\n{route1_end['type'].title()}\n\nFrom: {route1_start['name']}\n{route1_miles} mi • {route1_time}"},
            # Route 2 markers
            {"lon": route2_start["lng"], "lat": route2_start["lat"], 
             "type": "start", "route": "Route B", "color": [34, 197, 94, 255],
             "name": route2_start["name"],
             "info": f"🟢 START\n{route2_start['name']}\n{route2_start['type'].title()}\n\n→ {route2_end['name']}\n{route2_miles} mi • {route2_time}"},
            {"lon": route2_end["lng"], "lat": route2_end["lat"], 
             "type": "end", "route": "Route B", "color": [34, 197, 94, 255],
             "name": route2_end["name"],
             "info": f"🟢 END\n{route2_end['name']}\n{route2_end['type'].title()}\n\nFrom: {route2_start['name']}\n{route2_miles} mi • {route2_time}"},
            # Route 3 markers
            {"lon": route3_start["lng"], "lat": route3_start["lat"], 
             "type": "start", "route": "Route C", "color": [245, 159, 66, 255],
             "name": route3_start["name"],
             "info": f"🟠 START\n{route3_start['name']}\n{route3_start['type'].title()}\n\n→ {route3_end['name']}\n{route3_miles} mi • {route3_time}"},
            {"lon": route3_end["lng"], "lat": route3_end["lat"], 
             "type": "end", "route": "Route C", "color": [245, 159, 66, 255],
             "name": route3_end["name"],
             "info": f"🟠 END\n{route3_end['name']}\n{route3_end['type'].title()}\n\nFrom: {route3_start['name']}\n{route3_miles} mi • {route3_time}"},
        ])
        
        # Create scatterplot layer for markers
        markers_layer = pdk.Layer(
            "ScatterplotLayer",
            data=markers_data,
            get_position=["lon", "lat"],
            get_fill_color="color",
            get_radius=1000,
            radius_min_pixels=7,
            radius_max_pixels=10,
            pickable=True,
            auto_highlight=True,
            stroked=True,
            get_line_color=[255, 255, 255, 255],
            line_width_min_pixels=2,
        )
        
        # Create text labels for landmarks with better visibility
        text_data = pd.DataFrame([
            {"lon": route1_start["lng"], "lat": route1_start["lat"], "text": route1_start["name"], "color": [255, 255, 255, 255]},
            {"lon": route1_end["lng"], "lat": route1_end["lat"], "text": route1_end["name"], "color": [255, 255, 255, 255]},
            {"lon": route2_start["lng"], "lat": route2_start["lat"], "text": route2_start["name"], "color": [255, 255, 255, 255]},
            {"lon": route2_end["lng"], "lat": route2_end["lat"], "text": route2_end["name"], "color": [255, 255, 255, 255]},
            {"lon": route3_start["lng"], "lat": route3_start["lat"], "text": route3_start["name"], "color": [255, 255, 255, 255]},
            {"lon": route3_end["lng"], "lat": route3_end["lat"], "text": route3_end["name"], "color": [255, 255, 255, 255]},
        ])
        
        text_layer = pdk.Layer(
            "TextLayer",
            data=text_data,
            get_position=["lon", "lat"],
            get_text="text",
            get_color="color",
            get_size=16,
            get_angle=0,
            get_text_anchor='"middle"',
            get_alignment_baseline='"bottom"',
            get_pixel_offset=[0, -18],
            background=True,
            get_background_color=[10, 20, 40, 220],
            background_padding=[6, 3, 6, 3],
            font_family='"Arial", sans-serif',
            font_weight=700,
            pickable=False,
        )
        
        # Set view centered on the average of all routes
        all_lats = [p[1] for route in [route1_path, route2_path, route3_path] for p in route]
        all_lngs = [p[0] for route in [route1_path, route2_path, route3_path] for p in route]
        center_lat = sum(all_lats) / len(all_lats)
        center_lng = sum(all_lngs) / len(all_lngs)
        
        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lng,
            zoom=10,
            bearing=0,
            pitch=45
        )
        
        # Create deck with custom tooltip
        deck = pdk.Deck(
            layers=[path_layer, markers_layer, text_layer],
            initial_view_state=view_state,
            tooltip={
                "html": "<b>{info}</b>",
                "style": {
                    "backgroundColor": "rgba(0, 0, 0, 0.9)",
                    "color": "white",
                    "fontSize": "13px",
                    "padding": "12px",
                    "borderRadius": "8px",
                    "fontFamily": "Arial, sans-serif",
                    "whiteSpace": "pre-line",
                    "border": "2px solid rgba(96, 165, 250, 0.5)"
                }
            },
            map_style="mapbox://styles/mapbox/dark-v10"
        )
        
        st.pydeck_chart(deck, use_container_width=True, height=340)
        
        # Additional JavaScript to connect map panel and pydeck seamlessly
        st.markdown(
            """
            <script>
            // Connect map panel and pydeck chart seamlessly
            function connectMapToPanel() {
                const mapPanels = document.querySelectorAll('.map-panel');
                mapPanels.forEach(panel => {
                    const mapColumn = panel.closest('[data-testid="column"]');
                    if (!mapColumn) return;
                    
                    // Remove all gaps in vertical block
                    const parentDiv = panel.closest('[data-testid="stVerticalBlock"]');
                    if (parentDiv && mapColumn.contains(parentDiv)) {
                        parentDiv.style.gap = '0px';
                        parentDiv.style.margin = '0px';
                        parentDiv.style.padding = '0px';
                        
                        // Set all children to have no margin/padding
                        Array.from(parentDiv.children).forEach(child => {
                            child.style.margin = '0px';
                            child.style.padding = '0px';
                        });
                    }
                    
                    // Find pydeck chart and remove all spacing
                    const deckChart = mapColumn.querySelector('[data-testid="stDeckGlJsonChart"]');
                    if (deckChart) {
                        deckChart.style.marginTop = '0px';
                        deckChart.style.paddingTop = '0px';
                        
                        // Traverse up and remove spacing from all parents
                        let parent = deckChart.parentElement;
                        while (parent && parent !== mapColumn) {
                            parent.style.marginTop = '0px';
                            parent.style.paddingTop = '0px';
                            parent = parent.parentElement;
                        }
                    }
                });
            }
            
            // Run immediately and multiple times
            connectMapToPanel();
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', connectMapToPanel);
            }
            setTimeout(connectMapToPanel, 100);
            setTimeout(connectMapToPanel, 300);
            setTimeout(connectMapToPanel, 500);
            setTimeout(connectMapToPanel, 1000);
            </script>
            """,
            unsafe_allow_html=True,
        )


def render_footer() -> None:
    st.markdown(
        """
        <div class="footer">© 2025 Tech Mahindra. All rights reserved.</div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    set_page_config()
    inject_css()
    inject_js()
    
    render_top_nav()
    render_showcase()
    render_footer()


if __name__ == "__main__":
    main()
