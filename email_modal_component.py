"""
email_modal_component.py

Professional modal dialog component for email confirmation.
"""

import streamlit as st
from email_service import send_vehicle_alert_email

@st.dialog("Email Preview", width="large")
def render_email_confirmation_modal():
    """
    Render email confirmation modal using @st.dialog decorator.
    This creates a true modal dialog overlay showing the actual email content.
    """
    # Check if any email confirmation is triggered
    modal_keys = [key for key in st.session_state.keys() if key.startswith("show_email_confirm_")]
    
    if not modal_keys:
        st.info("No email confirmation pending.")
        return
    
    # Get the first triggered modal
    modal_key = modal_keys[0]
    email_key = modal_key.replace("show_email_confirm_", "")
    vehicle_data = st.session_state.get(f"email_data_{email_key}", {})
    
    if not vehicle_data:
        st.error("Vehicle data not found.")
        return
    
    # No header - start directly with content
    
    # Generate email content using existing LLM
    try:
        from email_service import EmailGenerator, VehicleAlert
        email_generator = EmailGenerator()
        
        # Create VehicleAlert object from vehicle_data
        alert = VehicleAlert(
            vin=vehicle_data.get('vin', ''),
            model=vehicle_data.get('model', ''),
            primary_failed_part=vehicle_data.get('primary_failed_part', ''),
            risk_percentage=vehicle_data.get('pred_prob_pct', 0),
            mileage=str(vehicle_data.get('mileage', vehicle_data.get('mileage_bucket', 'N/A'))),
            age=str(vehicle_data.get('age', vehicle_data.get('age_bucket', 'N/A'))),
            timestamp=str(vehicle_data.get('timestamp', '')),
            customer_email=vehicle_data.get('customer_email', ''),
            customer_mobile=vehicle_data.get('customer_mobile', ''),
            dealer_name=vehicle_data.get('dealer_name', ''),
            dealer_distance_km=vehicle_data.get('dealer_distance_km', 0),
            city=vehicle_data.get('city', ''),
            failure_description=vehicle_data.get('failure_description', 'High risk of component failure')
        )
        
        # Generate email content
        email_data = email_generator.generate_vehicle_alert_email(alert)
        email_content = email_data['body']
        
        # Display the generated email content - full width
        st.markdown("### Email Content Preview")
        
        # Email header info - optimized for no wrapping
        col1, col2, col3, col4 = st.columns([2, 3, 1, 1])
        with col1:
            st.markdown(f"**To:** {vehicle_data.get('customer_email', 'N/A')}")
        with col2:
            st.markdown(f"**Subject:** {email_data.get('subject', 'Vehicle Maintenance Alert')}")
        with col3:
            st.markdown(f"**Risk:** {vehicle_data.get('pred_prob_pct', 0):.1f}%")
        with col4:
            st.markdown(f"**Model:** {vehicle_data.get('model', 'N/A')}")
        
        st.markdown("---")
        
        # Display the email content in a styled container - no text wrapping
        st.markdown("""
        <div style="
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 20px;
            margin: 8px 0;
            box-shadow: 0 1px 4px rgba(0,0,0,0.1);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            font-size: 14px;
            max-height: 500px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
            min-width: 800px;
        ">
        """, unsafe_allow_html=True)
        
        # Render the email content
        st.markdown(email_content, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Failed to generate email content: {str(e)}")
        # Fallback to basic vehicle info
        st.markdown("### Vehicle Information")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Model:** {vehicle_data.get('model', 'N/A')}")
            st.markdown(f"**VIN:** {vehicle_data.get('vin', 'N/A')[:8]}...")
            st.markdown(f"**Failed Part:** {vehicle_data.get('primary_failed_part', 'N/A')}")
        with col2:
            st.markdown(f"**Risk Level:** {vehicle_data.get('pred_prob_pct', 0):.1f}%")
            st.markdown(f"**Customer:** {vehicle_data.get('customer_email', 'N/A')}")
            st.markdown(f"**Dealer:** {vehicle_data.get('dealer_name', 'N/A')}")
    
    # Warning message removed as requested
    
    # Action buttons - compact
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    col_btn1, col_btn2 = st.columns(2, gap="medium")
    
    with col_btn1:
        if st.button("Send Email", key=f"modal_send_{email_key}", type="primary", use_container_width=True):
            # Send email using the email service
            result = send_vehicle_alert_email(vehicle_data)
            
            if result['success']:
                st.success("Email sent successfully!")
                st.balloons()
                # Clear the confirmation state
                st.session_state[f"show_email_confirm_{email_key}"] = False
                st.rerun()
            else:
                st.error(f"Failed to send email: {result.get('message', 'Unknown error')}")
    
    with col_btn2:
        if st.button("Cancel", key=f"modal_cancel_{email_key}", use_container_width=True):
            st.info("Email sending cancelled")
            # Clear the confirmation state
            st.session_state[f"show_email_confirm_{email_key}"] = False
            st.rerun()
