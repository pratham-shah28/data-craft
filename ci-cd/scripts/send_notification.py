#!/usr/bin/env python3
"""
Send Email Notifications
Sends email notifications for CI/CD pipeline events
"""

import sys
import os
import json
import yaml
import argparse
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Add paths
project_root = Path(__file__).parent.parent.parent
config_path = project_root / "ci-cd" / "config" / "ci_cd_config.yaml"
outputs_dir = project_root / "outputs"

def load_config():
    """Load CI/CD configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_pipeline_summary() -> Dict:
    """Get summary of pipeline execution"""
    summary = {
        "status": "unknown",
        "best_model": "unknown",
        "metrics": {}
    }
    
    # Try to load pipeline status
    status_file = outputs_dir / "pipeline_status.json"
    if status_file.exists():
        with open(status_file, 'r') as f:
            status_data = json.load(f)
            summary["status"] = status_data.get("status", "unknown")
            summary["best_model"] = status_data.get("best_model", "unknown")
            summary["metrics"]["accuracy"] = status_data.get("accuracy", 0)
    
    # Try to load validation report
    validation_file = outputs_dir / "validation" / "validation_report.json"
    if validation_file.exists():
        with open(validation_file, 'r') as f:
            validation_data = json.load(f)
            summary["validation_status"] = validation_data.get("status", "unknown")
            summary["metrics"].update(validation_data.get("metrics", {}))
    
    # Try to load push report
    push_file = outputs_dir / "validation" / "push_report.json"
    if push_file.exists():
        with open(push_file, 'r') as f:
            push_data = json.load(f)
            summary["registry_path"] = push_data.get("package_path", "")
    
    return summary

def create_email_body(status: str, summary: Dict, workflow_url: Optional[str] = None) -> str:
    """Create email body content"""
    status_emoji = "✅" if status == "success" else "❌"
    status_text = "SUCCESS" if status == "success" else "FAILED"
    
    body = f"""
{status_emoji} Model Training CI/CD Pipeline - {status_text}

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'=' * 60}
PIPELINE SUMMARY
{'=' * 60}

Status: {summary.get('status', 'unknown')}
Best Model: {summary.get('best_model', 'unknown')}

Metrics:
"""
    
    metrics = summary.get('metrics', {})
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                body += f"  • {key.replace('_', ' ').title()}: {value:.2f}\n"
            else:
                body += f"  • {key.replace('_', ' ').title()}: {value}\n"
    
    if summary.get('validation_status'):
        body += f"\nValidation: {summary.get('validation_status', 'unknown')}\n"
    
    if summary.get('registry_path'):
        body += f"\nModel Registry: {summary.get('registry_path', '')}\n"
    
    if workflow_url:
        body += f"\nWorkflow Run: {workflow_url}\n"
    
    body += f"""
{'=' * 60}

This is an automated notification from the Model Training CI/CD Pipeline.
"""
    
    return body

def send_email(config: Dict, status: str, workflow_url: Optional[str] = None):
    """Send email notification
    
    Email Configuration:
    - SMTP server, port, from_email, to_email: From config file
    - EMAIL_SMTP_USER: Environment variable (optional, defaults to from_email in config)
    - EMAIL_SMTP_PASSWORD: Environment variable (REQUIRED)
    
    For local execution: Set EMAIL_SMTP_PASSWORD as env var
    For GitHub Actions: Set EMAIL_SMTP_PASSWORD as GitHub secret
    """
    email_config = config['notifications']['email']
    
    # Get email credentials
    # SMTP user: env var EMAIL_SMTP_USER, or fallback to from_email in config
    smtp_user = os.environ.get('EMAIL_SMTP_USER', email_config.get('from_email'))
    # SMTP password: MUST be set as environment variable (for security)
    smtp_password = os.environ.get('EMAIL_SMTP_PASSWORD')
    
    if not smtp_password:
        print("⚠ Email password not set - skipping email notification")
        print("   Set EMAIL_SMTP_PASSWORD environment variable to enable email notifications")
        return
    
    summary = get_pipeline_summary()
    body = create_email_body(status, summary, workflow_url)
    
    # Create message
    msg = MIMEMultipart()
    msg['From'] = email_config['from_email']
    msg['To'] = email_config['to_email']
    msg['Subject'] = f"Model Training CI/CD: {status.upper()}"
    
    msg.attach(MIMEText(body, 'plain'))
    
    # Send email
    try:
        server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
        if email_config.get('use_tls', True):
            server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        server.quit()
        
        print(f"✓ Email sent to {email_config['to_email']}")
        
    except Exception as e:
        print(f"⚠ Failed to send email: {e}")

def main():
    """Main notification function"""
    parser = argparse.ArgumentParser(description='Send email notification')
    parser.add_argument('--status', required=True, choices=['success', 'failure', 'cancelled'],
                       help='Pipeline status')
    parser.add_argument('--workflow-run-url', help='GitHub workflow run URL')
    args = parser.parse_args()
    
    print("=" * 70)
    print("SEND NOTIFICATION")
    print("=" * 70)
    
    try:
        config = load_config()
        print("✓ Loaded configuration")
        
        send_email(config, args.status, args.workflow_run_url)
        
        print("\n" + "=" * 70)
        print("NOTIFICATION SENT")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\n⚠ NOTIFICATION ERROR: {str(e)}")
        # Don't fail the pipeline if notification fails
        return 0

if __name__ == "__main__":
    sys.exit(main())

