"""
Enhanced Real-Time Vehicle Feed table with Email and SMS action buttons.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any
import base64
from io import StringIO
from email_service import show_email_confirmation, send_vehicle_alert_email

def create_action_buttons_html(pred_prob_pct: float, row_index: int) -> str:
    """Create HTML for Email and SMS action buttons based on prediction percentage."""
    if pred_prob_pct > 50:
        # Create unique keys for each button to avoid conflicts
        email_key = f"email_btn_{row_index}"
        sms_key = f"sms_btn_{row_index}"
        
        return f"""
        <div style="display: flex; gap: 8px; justify-content: center;">
            <button onclick="handleEmailAction({row_index})" 
                    style="
                        background: #3b82f6; 
                        color: white; 
                        border: none; 
                        padding: 6px 12px; 
                        border-radius: 6px; 
                        font-size: 12px; 
                        font-weight: 500;
                        cursor: pointer;
                        transition: background-color 0.2s;
                    "
                    onmouseover="this.style.backgroundColor='#2563eb'"
                    onmouseout="this.style.backgroundColor='#3b82f6'">
                📧 Email
            </button>
            <button onclick="handleSMSAction({row_index})" 
                    style="
                        background: #10b981; 
                        color: white; 
                        border: none; 
                        padding: 6px 12px; 
                        border-radius: 6px; 
                        font-size: 12px; 
                        font-weight: 500;
                        cursor: pointer;
                        transition: background-color 0.2s;
                    "
                    onmouseover="this.style.backgroundColor='#059669'"
                    onmouseout="this.style.backgroundColor='#10b981'">
                📱 SMS
            </button>
        </div>
        """
    else:
        return '<div style="color: #6b7280; font-size: 12px; text-align: center;">No action needed</div>'

def create_enhanced_dataframe(df_show: pd.DataFrame) -> pd.DataFrame:
    """Create enhanced dataframe with action buttons column."""
    df_enhanced = df_show.copy()
    
    # Add action buttons column
    action_buttons = []
    for idx, row in df_enhanced.iterrows():
        pred_prob_pct = row.get('pred_prob_pct', 0)
        action_html = create_action_buttons_html(pred_prob_pct, idx)
        action_buttons.append(action_html)
    
    df_enhanced['Actions'] = action_buttons
    
    # Reorder columns to put Actions at the end
    columns = [col for col in df_enhanced.columns if col != 'Actions']
    columns.append('Actions')
    df_enhanced = df_enhanced[columns]
    
    return df_enhanced

def render_enhanced_inference_table(df_log: pd.DataFrame, date_range: tuple, text_filter: str, rows_to_show: int):
    """Render the enhanced Real-Time Vehicle Feed table with integrated action buttons."""
    
    # Apply filters
    dr_start, dr_end = date_range
    mask = (df_log["timestamp"].dt.date >= dr_start) & (df_log["timestamp"].dt.date <= dr_end)
    if text_filter.strip():
        t = text_filter.strip().lower()
        mask &= df_log["model"].str.lower().str.contains(t) | df_log["primary_failed_part"].str.lower().str.contains(t)
    
    df_show = df_log[mask].sort_values("timestamp", ascending=False).head(rows_to_show)
    
    if df_show.empty:
        st.markdown("<div style='padding:12px; color:#94a3b8; text-align: center;'>No data found for the selected filters.</div>", unsafe_allow_html=True)
        return
    
    # Define column renaming
    column_renames = {
        "timestamp": "Event Timestamp",
        "model": "Model",
        "primary_failed_part": "Primary Failed Part",
        "mileage": "Mileage",
        "mileage_bucket": "Mileage",
        "age": "Age",
        "age_bucket": "Age",
        "pred_prob_pct": "Predictive %"
    }
    
    # Rename columns that exist
    df_display = df_show.rename(columns=column_renames)
    
    # Hide pred_prob column if it exists
    columns_to_display = [col for col in df_display.columns if col != "pred_prob"]
    df_display = df_display[columns_to_display]
    
    # Display the table with integrated action buttons - no spacing div
    # Add table headers with minimal spacing
    header_col1, header_col2, header_col3, header_col4, header_col5, header_col6, header_col7, header_col8 = st.columns([2, 1.5, 1.5, 1, 1, 1, 1, 1])
    
    with header_col1:
        st.markdown('<div style="margin: 0; padding: 2px 0;"><strong>Event Timestamp</strong></div>', unsafe_allow_html=True)
    with header_col2:
        st.markdown('<div style="margin: 0; padding: 2px 0;"><strong>Model</strong></div>', unsafe_allow_html=True)
    with header_col3:
        st.markdown('<div style="margin: 0; padding: 2px 0;"><strong>Primary Failed Part</strong></div>', unsafe_allow_html=True)
    with header_col4:
        st.markdown('<div style="margin: 0; padding: 2px 0;"><strong>Mileage</strong></div>', unsafe_allow_html=True)
    with header_col5:
        st.markdown('<div style="margin: 0; padding: 2px 0;"><strong>Age</strong></div>', unsafe_allow_html=True)
    with header_col6:
        st.markdown('<div style="margin: 0; padding: 2px 0;"><strong>Predictive %</strong></div>', unsafe_allow_html=True)
    with header_col7:
        st.markdown('<div style="margin: 0; padding: 2px 0;"><strong>Email</strong></div>', unsafe_allow_html=True)
    with header_col8:
        st.markdown('<div style="margin: 0; padding: 2px 0;"><strong>SMS</strong></div>', unsafe_allow_html=True)
    
    # Add separator line with minimal spacing
    st.markdown('<div style="margin: 0; padding: 1px 0;"><hr style="border: 1px solid #374151; margin: 0;"></div>', unsafe_allow_html=True)
    
    # Create action buttons for each row with reduced spacing
    for idx, (_, row) in enumerate(df_show.iterrows()):
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([2, 1.5, 1.5, 1, 1, 1, 1, 1])
        
        with col1:
            st.markdown(f'<div style="margin: 0; padding: 1px 0; font-size: 13px;">{row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")}</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div style="margin: 0; padding: 1px 0; font-size: 13px;">{row["model"]}</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'<div style="margin: 0; padding: 1px 0; font-size: 13px;">{row["primary_failed_part"]}</div>', unsafe_allow_html=True)
        
        with col4:
            # Handle both 'mileage' and 'mileage_bucket' columns
            mileage_value = row.get('mileage', row.get('mileage_bucket', 'N/A'))
            if isinstance(mileage_value, (int, float)):
                st.markdown(f'<div style="margin: 0; padding: 1px 0; font-size: 13px;">{mileage_value:,.0f}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="margin: 0; padding: 1px 0; font-size: 13px;">{mileage_value}</div>', unsafe_allow_html=True)
        
        with col5:
            # Handle both 'age' and 'age_bucket' columns
            age_value = row.get('age', row.get('age_bucket', 'N/A'))
            if isinstance(age_value, (int, float)):
                st.markdown(f'<div style="margin: 0; padding: 1px 0; font-size: 13px;">{age_value:.1f}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="margin: 0; padding: 1px 0; font-size: 13px;">{age_value}</div>', unsafe_allow_html=True)
        
        with col6:
            # Color code the predictive percentage
            pred_pct = row['pred_prob_pct']
            if pred_pct > 50:
                st.markdown(f'<div style="margin: 0; padding: 1px 0; font-size: 13px;"><span style="color: #dc2626; font-weight: bold;">{pred_pct:.1f}%</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="margin: 0; padding: 1px 0; font-size: 13px;">{pred_pct:.1f}%</div>', unsafe_allow_html=True)
        
        with col7:
            # Email button - only show for high-risk vehicles
            if pred_pct > 50:
                # Create unique key for this email button
                email_key = f"email_{idx}_{row['timestamp']}"
                
                if st.button("Email", key=email_key, type="primary"):
                    # Store vehicle data in session state for confirmation
                    vehicle_data = {
                        'vin': row.get('vin', ''),
                        'model': row['model'],
                        'primary_failed_part': row['primary_failed_part'],
                        'pred_prob_pct': row['pred_prob_pct'],
                        'mileage': row.get('mileage', row.get('mileage_bucket', 'N/A')),
                        'age': row.get('age', row.get('age_bucket', 'N/A')),
                        'timestamp': str(row['timestamp']),
                        'customer_email': row.get('customer_email', 'bijus.digital.world@gmail.com'),
                        'customer_mobile': row.get('customer_mobile', '+91 9380636750'),
                        'dealer_name': row.get('dealer_name', 'Nearest Nissan Dealer'),
                        'dealer_distance_km': row.get('dealer_distance_km', 0),
                        'city': row.get('city', 'Unknown'),
                        'failure_description': row.get('failure_description', 'High risk of component failure')
                    }
                    
                    # Store data and trigger confirmation
                    st.session_state[f"email_data_{email_key}"] = vehicle_data
                    st.session_state[f"show_email_confirm_{email_key}"] = True
                    st.rerun()
                
                # Modal confirmation is handled at app level
            else:
                st.markdown('<div style="margin: 0; padding: 1px 0; font-size: 13px; text-align: center;">—</div>', unsafe_allow_html=True)
        
        with col8:
            # SMS button - only show for high-risk vehicles
            if pred_pct > 50:
                if st.button("SMS", key=f"sms_{idx}_{row['timestamp']}", type="secondary"):
                    sms_text = f"URGENT: {row['model']} vehicle has {row['pred_prob_pct']:.1f}% failure risk for {row['primary_failed_part']}. Immediate service required. Contact owner ASAP."
                    
                    # Copy to clipboard
                    import pyperclip
                    try:
                        pyperclip.copy(sms_text)
                        st.success("SMS text copied to clipboard!")
                    except:
                        st.info("SMS text ready to copy")
            else:
                st.markdown('<div style="margin: 0; padding: 1px 0; font-size: 13px; text-align: center;">—</div>', unsafe_allow_html=True)
    
    # Add summary statistics
    high_risk_count = len(df_show[df_show['pred_prob_pct'] > 50])
    total_count = len(df_show)
    
    if high_risk_count > 0:
        st.markdown(f"""
        <div style="
            background: #fef2f2; 
            border: 1px solid #fecaca; 
            border-radius: 8px; 
            padding: 12px; 
            margin-top: 12px;
        ">
            <div style="color: #dc2626; font-weight: 600; margin-bottom: 4px;">
                High Risk Alert Summary
            </div>
            <div style="color: #374151; font-size: 14px;">
                {high_risk_count} out of {total_count} vehicles require immediate attention (Predictive % > 50%)
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("No high-risk vehicles detected. All vehicles are within normal parameters.")

def render_action_buttons_legacy(df_show: pd.DataFrame):
    """Alternative implementation using Streamlit buttons (if HTML doesn't work)."""
    st.markdown("### Action Buttons for High-Risk Vehicles")
    
    # Filter high-risk vehicles
    high_risk_df = df_show[df_show['pred_prob_pct'] > 50]
    
    if high_risk_df.empty:
        st.info("No high-risk vehicles found (Predictive % > 50%)")
        return
    
    st.markdown(f"**Found {len(high_risk_df)} high-risk vehicles requiring attention:**")
    
    for idx, (_, row) in enumerate(high_risk_df.iterrows()):
        with st.expander(f"Vehicle {idx + 1}: {row['model']} - {row['primary_failed_part']} ({row['pred_prob_pct']:.1f}%)"):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**Model:** {row['model']}")
                st.write(f"**Part:** {row['primary_failed_part']}")
                st.write(f"**Risk Level:** {row['pred_prob_pct']:.1f}%")
                st.write(f"**Timestamp:** {row['timestamp']}")
            
            with col2:
                if st.button(f"📧 Email", key=f"email_{idx}"):
                    # Create email content
                    subject = f"High Risk Alert - {row['model']} Vehicle"
                    body = f"""Vehicle Model: {row['model']}
Predicted Failed Part: {row['primary_failed_part']}
Risk Level: {row['pred_prob_pct']:.1f}%
Timestamp: {row['timestamp']}

This vehicle has a high probability of failure and requires immediate attention."""
                    
                    # Copy to clipboard or show
                    st.code(body, language="text")
                    st.success("Email content ready to copy!")
            
            with col3:
                if st.button(f"📱 SMS", key=f"sms_{idx}"):
                    sms_text = f"ALERT: {row['model']} vehicle has {row['pred_prob_pct']:.1f}% failure risk for {row['primary_failed_part']}. Immediate service recommended."
                    st.code(sms_text, language="text")
                    st.success("SMS text ready to copy!")

def create_email_template(model: str, part: str, pred_prob: float, timestamp: str) -> str:
    """Create a professional email template for high-risk alerts."""
    return f"""
Subject: URGENT: High Risk Vehicle Alert - {model}

Dear Service Team,

This is an automated alert regarding a high-risk vehicle that requires immediate attention.

VEHICLE DETAILS:
• Model: {model}
• Predicted Failed Part: {part}
• Risk Probability: {pred_prob:.1f}%
• Alert Time: {timestamp}

RECOMMENDED ACTIONS:
1. Contact the vehicle owner immediately
2. Schedule urgent service appointment
3. Prepare replacement parts inventory
4. Assign priority technician

This vehicle exceeds our 50% risk threshold and should be treated as a priority case.

Best regards,
Vehicle Predictive Insights System
    """.strip()

def create_sms_template(model: str, part: str, pred_prob: float) -> str:
    """Create a concise SMS template for high-risk alerts."""
    return f"URGENT: {model} vehicle has {pred_prob:.1f}% failure risk for {part}. Immediate service required. Contact owner ASAP."
