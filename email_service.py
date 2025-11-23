"""
email_service.py

Professional email generation and sending service using AWS Bedrock and SES.
Generates contextual emails for vehicle failure alerts with dealer information.
"""

import boto3
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import streamlit as st

from helper import km_to_miles

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VehicleAlert:
    """Data class for vehicle alert information"""
    vin: str
    model: str
    primary_failed_part: str
    risk_percentage: float
    mileage: str
    age: str
    timestamp: str
    customer_email: str
    customer_mobile: str
    dealer_name: str
    dealer_distance_km: float
    city: str
    failure_description: str

class EmailGenerator:
    """Generate professional emails using AWS Bedrock"""
    
    def __init__(self):
        # Use the same region as your existing Bedrock configuration
        self.bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
        
    def generate_vehicle_alert_email(self, alert: VehicleAlert) -> Dict[str, str]:
        """
        Generate a professional vehicle alert email using Bedrock AI
        """
        try:
            # Create the prompt for email generation
            prompt = self._create_email_prompt(alert)
            
            # Call Bedrock to generate email content (matching existing configuration)
            response = self.bedrock.invoke_model(
                modelId='anthropic.claude-3-haiku-20240307-v1:0',
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "temperature": 0.18,
                    "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                })
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            email_content = response_body['content'][0]['text']
            
            # Parse the structured response
            return self._parse_email_response(email_content, alert)
            
        except Exception as e:
            logger.error(f"Email generation failed: {e}")
            return self._create_fallback_email(alert)
    
    def _create_email_prompt(self, alert: VehicleAlert) -> str:
        """Create a detailed prompt for email generation"""
        distance_miles = km_to_miles(alert.dealer_distance_km)
        distance_text = f"{distance_miles:.1f} mi" if distance_miles is not None else "N/A"
        return f"""
You are a professional automotive service advisor writing an urgent vehicle maintenance alert email.

VEHICLE INFORMATION:
- VIN: {alert.vin}
- Model: {alert.model}
- Failed Part: {alert.primary_failed_part}
- Risk Level: {alert.risk_percentage:.1f}%
- Mileage: {alert.mileage}
- Vehicle Age: {alert.age}
- Location: {alert.city}
- Failure Description: {alert.failure_description}

CUSTOMER INFORMATION:
- Email: {alert.customer_email}
- Mobile: {alert.customer_mobile}

DEALER INFORMATION:
- Nearest Dealer: {alert.dealer_name}
- Distance: {distance_text}
- Location: {alert.city}

Please generate a professional email with the following structure:

SUBJECT: [Create an urgent, professional subject line]

BODY:
1. Professional greeting
2. Urgency explanation (why immediate action is needed)
3. Specific issue details and potential consequences
4. Recommended actions
5. Dealer contact information and next steps
6. Professional closing

TONE: Urgent but professional, empathetic, solution-focused
LENGTH: 150-200 words
FORMAT: Professional business email

Return the response in this exact JSON format:
{{
    "subject": "Subject line here",
    "body": "Complete email body here",
    "urgency_level": "high/medium/low",
    "recommended_actions": ["action1", "action2", "action3"]
}}
"""
    
    def _parse_email_response(self, content: str, alert: VehicleAlert) -> Dict[str, str]:
        """Parse the AI-generated email response"""
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                email_data = json.loads(json_match.group())
                return {
                    "subject": email_data.get("subject", f"Urgent Vehicle Service Required - {alert.model}"),
                    "body": email_data.get("body", self._create_fallback_email(alert)["body"]),
                    "urgency_level": email_data.get("urgency_level", "high"),
                    "recommended_actions": email_data.get("recommended_actions", ["Contact dealer immediately", "Schedule service appointment"])
                }
        except:
            pass
        
        # Fallback if JSON parsing fails
        return self._create_fallback_email(alert)
    
    def _create_fallback_email(self, alert: VehicleAlert) -> Dict[str, str]:
        """Create a fallback email if AI generation fails"""
        subject = f"URGENT: Vehicle Service Required - {alert.model} ({alert.vin[:8]}...)"
        distance_miles = km_to_miles(alert.dealer_distance_km)
        distance_text = f"{distance_miles:.1f} mi" if distance_miles is not None else "N/A"

        body = f"""
Dear Valued Customer,

We have identified a critical issue with your {alert.model} vehicle that requires immediate attention.

VEHICLE DETAILS:
• VIN: {alert.vin}
• Model: {alert.model}
• Issue: {alert.primary_failed_part}
• Risk Level: {alert.risk_percentage:.1f}%
• Current Location: {alert.city}

URGENT ACTION REQUIRED:
Your vehicle has a {alert.risk_percentage:.1f}% probability of failure related to the {alert.primary_failed_part}. Immediate service is recommended to prevent potential breakdown or safety issues.

RECOMMENDED ACTIONS:
1. Contact your nearest dealer immediately
2. Schedule an urgent service appointment
3. Avoid long-distance driving until service is completed
4. Keep emergency contact information handy

NEAREST SERVICE CENTER:
• Dealer: {alert.dealer_name}
• Distance: {distance_text} from your location
• Location: {alert.city}

Please contact us immediately to schedule service. Your safety is our priority.

Best regards,
Nissan Customer Service Team
        """.strip()
        
        return {
            "subject": subject,
            "body": body,
            "urgency_level": "high",
            "recommended_actions": ["Contact dealer immediately", "Schedule service appointment", "Avoid long-distance driving"]
        }

class EmailSender:
    """Send emails using AWS SES"""
    
    def __init__(self):
        self.ses = boto3.client('ses', region_name='us-east-1')
        self.sender_email = "noreply@nissan-telematics.com"  # Configure this in SES
        
    def send_vehicle_alert_email(self, alert: VehicleAlert, email_content: Dict[str, str]) -> bool:
        """
        Send the generated email via AWS SES
        """
        try:
            # Send email
            response = self.ses.send_email(
                Source=self.sender_email,
                Destination={
                    'ToAddresses': [alert.customer_email]
                },
                Message={
                    'Subject': {
                        'Data': email_content['subject'],
                        'Charset': 'UTF-8'
                    },
                    'Body': {
                        'Text': {
                            'Data': email_content['body'],
                            'Charset': 'UTF-8'
                        },
                        'Html': {
                            'Data': self._create_html_email(email_content['body']),
                            'Charset': 'UTF-8'
                        }
                    }
                }
            )
            
            logger.info(f"Email sent successfully. MessageId: {response['MessageId']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def _create_html_email(self, text_body: str) -> str:
        """Convert text email to HTML format"""
        html_body = text_body.replace('\n', '<br>')
        html_body = html_body.replace('•', '&bull;')
        
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                {html_body}
            </div>
        </body>
        </html>
        """

class VehicleEmailService:
    """Main service class for vehicle email alerts"""
    
    def __init__(self):
        self.email_generator = EmailGenerator()
        self.email_sender = EmailSender()
    
    def process_vehicle_alert(self, vehicle_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a vehicle alert and send email
        """
        try:
            # Create VehicleAlert object
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
                failure_description=vehicle_data.get('failure_description', '')
            )
            
            # Generate email content
            email_content = self.email_generator.generate_vehicle_alert_email(alert)
            
            # Send email
            email_sent = self.email_sender.send_vehicle_alert_email(alert, email_content)
            
            return {
                "success": email_sent,
                "email_content": email_content,
                "message": "Email sent successfully" if email_sent else "Failed to send email"
            }
            
        except Exception as e:
            logger.error(f"Vehicle alert processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to process vehicle alert"
            }

# Streamlit integration functions
def show_email_confirmation(vehicle_data: Dict[str, Any]) -> bool:
    """Show email confirmation dialog in Streamlit"""
    st.markdown("### 📧 Email Confirmation")
    st.markdown(f"**Vehicle:** {vehicle_data.get('model', 'N/A')} ({vehicle_data.get('vin', 'N/A')[:8]}...)")
    st.markdown(f"**Risk Level:** {vehicle_data.get('pred_prob_pct', 0):.1f}%")
    st.markdown(f"**Customer:** {vehicle_data.get('customer_email', 'N/A')}")
    st.markdown(f"**Nearest Dealer:** {vehicle_data.get('dealer_name', 'N/A')}")
    
    # Create a unique key for this confirmation dialog
    confirm_key = f"email_confirm_{vehicle_data.get('vin', 'unknown')}_{vehicle_data.get('timestamp', 'unknown')}"
    
    col1, col2 = st.columns(2)
    with col1:
        send_email = st.button("✅ Send Email", key=f"{confirm_key}_send", type="primary")
    with col2:
        cancel_email = st.button("❌ Cancel", key=f"{confirm_key}_cancel")
    
    if send_email:
        return True
    elif cancel_email:
        return False
    
    return False

def send_vehicle_alert_email(vehicle_data: Dict[str, Any]) -> Dict[str, Any]:
    """Main function to send vehicle alert email"""
    email_service = VehicleEmailService()
    return email_service.process_vehicle_alert(vehicle_data)
