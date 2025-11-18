#!/usr/bin/env python3
"""
Email Notification Script
Sends email notifications for CI/CD pipeline events (email only, no Slack)

Usage:
    python send_notifications.py <event_type> [additional_args...]
    
Event types:
    success - Training completed successfully
    failed - Training failed
    validation_failed - Model validation failed
    bias_failed - Bias check failed
    comparison_failed - Model comparison failed (rollback recommended)
"""

import json
import sys
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime
import yaml

def load_config():
    """Load CI/CD configuration"""
    config_path = Path(__file__).parent.parent / "config" / "ci_cd_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def send_email_notification(
    smtp_server: str,
    smtp_port: int,
    sender: str,
    sender_password: str,
    recipients: list,
    subject: str,
    body: str
) -> bool:
    """
    Send email notification
    
    Args:
        smtp_server: SMTP server address
        smtp_port: SMTP port
        sender: Sender email address
        sender_password: Sender password or app password
        recipients: List of recipient email addresses
        subject: Email subject
        body: Email body text
        
    Returns:
        True if sent successfully, False otherwise
    """
    try:
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = ', '.join(recipients)
        
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender, sender_password)
            server.sendmail(sender, recipients, msg.as_string())
        
        print(f"✓ Email sent successfully to {', '.join(recipients)}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to send email: {e}")
        return False

def notify_training_complete(metrics_file: str):
    """Notify on training completion"""
    config = load_config()
    email_config = config['notifications']['email']
    
    if not email_config.get('enabled', False):
        print("Email notifications disabled in configuration")
        return
    
    # Load metrics
    if not Path(metrics_file).exists():
        print(f"Warning: Metrics file not found: {metrics_file}")
        return
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    best_model = metrics.get('best_model', {})
    accuracy_metrics = metrics.get('accuracy', {})
    
    message = f"""
Model Training Pipeline - SUCCESS

Best Model Selected: {best_model.get('name', 'Unknown')}
Composite Score: {best_model.get('score', 0):.2f}/100
Performance Score: {best_model.get('performance', 0):.2f}/100
Bias Score: {best_model.get('bias', 0):.2f}/100
Overall Accuracy: {accuracy_metrics.get('overall_accuracy', 0):.1f}%

Models Evaluated: {metrics.get('models_evaluated', 0)}
  {', '.join(metrics.get('models_list', []))}

Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Outputs:
  Local Directory: {metrics.get('outputs', {}).get('local_directory', 'N/A')}
  Files Saved: {metrics.get('outputs', {}).get('files_saved', 0)}
"""
    
    sender = os.getenv(email_config['sender_env'])
    sender_password = os.getenv(email_config['sender_password_env'])
    recipients_str = os.getenv(email_config['recipients_env'], '')
    recipients = [r.strip() for r in recipients_str.split(',') if r.strip()]
    
    if not sender or not sender_password or not recipients:
        print("Email configuration incomplete. Check environment variables:")
        print(f"  {email_config['sender_env']}: {'Set' if sender else 'NOT SET'}")
        print(f"  {email_config['sender_password_env']}: {'Set' if sender_password else 'NOT SET'}")
        print(f"  {email_config['recipients_env']}: {'Set' if recipients else 'NOT SET'}")
        return
    
    send_email_notification(
        smtp_server=email_config['smtp_server'],
        smtp_port=email_config['smtp_port'],
        sender=sender,
        sender_password=sender_password,
        recipients=recipients,
        subject="✅ Model Training Complete - MLOps Pipeline",
        body=message
    )

def notify_training_failed(error_message: str):
    """Notify on training failure"""
    config = load_config()
    email_config = config['notifications']['email']
    
    if not email_config.get('enabled', False):
        return
    
    message = f"""
Model Training Pipeline - FAILED

Error: {error_message}

Pipeline failed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please check the GitHub Actions logs or Airflow logs for detailed error information.
"""
    
    sender = os.getenv(email_config['sender_env'])
    sender_password = os.getenv(email_config['sender_password_env'])
    recipients_str = os.getenv(email_config['recipients_env'], '')
    recipients = [r.strip() for r in recipients_str.split(',') if r.strip()]
    
    if sender and sender_password and recipients:
        send_email_notification(
            smtp_server=email_config['smtp_server'],
            smtp_port=email_config['smtp_port'],
            sender=sender,
            sender_password=sender_password,
            recipients=recipients,
            subject="❌ Model Training Failed - MLOps Pipeline",
            body=message
        )

def notify_validation_failed(failed_checks: list):
    """Notify on validation failure"""
    config = load_config()
    email_config = config['notifications']['email']
    
    if not email_config.get('enabled', False):
        return
    
    message = f"""
Model Training Pipeline - VALIDATION FAILED

Failed Validation Checks:
{chr(10).join(f'  - {check}' for check in failed_checks)}

Model did not meet quality thresholds.
Deployment has been blocked.

Pipeline failed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please review the validation results and adjust model training parameters or thresholds.
"""
    
    sender = os.getenv(email_config['sender_env'])
    sender_password = os.getenv(email_config['sender_password_env'])
    recipients_str = os.getenv(email_config['recipients_env'], '')
    recipients = [r.strip() for r in recipients_str.split(',') if r.strip()]
    
    if sender and sender_password and recipients:
        send_email_notification(
            smtp_server=email_config['smtp_server'],
            smtp_port=email_config['smtp_port'],
            sender=sender,
            sender_password=sender_password,
            recipients=recipients,
            subject="⚠️ Model Validation Failed - MLOps Pipeline",
            body=message
        )

def notify_bias_failed(bias_details: dict):
    """Notify on bias check failure"""
    config = load_config()
    email_config = config['notifications']['email']
    
    if not email_config.get('enabled', False):
        return
    
    failed_checks = [k for k, v in bias_details.items() if not v.get('passed', True)]
    
    message = f"""
Model Training Pipeline - BIAS CHECK FAILED

Failed Bias Checks:
{chr(10).join(f'  - {check.replace("_", " ").title()}: {bias_details[check].get("value", "N/A")}' for check in failed_checks)}

Bias thresholds exceeded acceptable limits.
Deployment has been blocked.

Pipeline failed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please review the bias detection results and consider model adjustments.
"""
    
    sender = os.getenv(email_config['sender_env'])
    sender_password = os.getenv(email_config['sender_password_env'])
    recipients_str = os.getenv(email_config['recipients_env'], '')
    recipients = [r.strip() for r in recipients_str.split(',') if r.strip()]
    
    if sender and sender_password and recipients:
        send_email_notification(
            smtp_server=email_config['smtp_server'],
            smtp_port=email_config['smtp_port'],
            sender=sender,
            sender_password=sender_password,
            recipients=recipients,
            subject="⚠️ Model Bias Check Failed - MLOps Pipeline",
            body=message
        )

def notify_comparison_failed(comparison_details: dict):
    """Notify on model comparison failure (rollback recommended)"""
    config = load_config()
    email_config = config['notifications']['email']
    
    if not email_config.get('enabled', False):
        return
    
    message = f"""
Model Training Pipeline - MODEL COMPARISON FAILED

New model performs worse than previous model.

Previous Model Score: {comparison_details.get('previous_score', 'N/A'):.2f}/100
Current Model Score:  {comparison_details.get('current_score', 'N/A'):.2f}/100
Improvement:          {comparison_details.get('improvement', 0):+.2f} ({comparison_details.get('improvement_pct', 0):+.2f}%)

Recommendation: Rollback to previous model version

Pipeline failed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    sender = os.getenv(email_config['sender_env'])
    sender_password = os.getenv(email_config['sender_password_env'])
    recipients_str = os.getenv(email_config['recipients_env'], '')
    recipients = [r.strip() for r in recipients_str.split(',') if r.strip()]
    
    if sender and sender_password and recipients:
        send_email_notification(
            smtp_server=email_config['smtp_server'],
            smtp_port=email_config['smtp_port'],
            sender=sender,
            sender_password=sender_password,
            recipients=recipients,
            subject="⚠️ Model Comparison Failed - Rollback Recommended",
            body=message
        )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python send_notifications.py <event_type> [args...]")
        print("\nEvent types:")
        print("  success <metrics_file>")
        print("  failed <error_message>")
        print("  validation_failed")
        print("  bias_failed")
        print("  comparison_failed")
        sys.exit(1)
    
    event_type = sys.argv[1]
    
    if event_type == "success":
        metrics_file = sys.argv[2] if len(sys.argv) > 2 else "outputs/model-training/pipeline_summary.json"
        notify_training_complete(metrics_file)
    elif event_type == "failed":
        error_message = sys.argv[2] if len(sys.argv) > 2 else "Unknown error"
        notify_training_failed(error_message)
    elif event_type == "validation_failed":
        # Could parse validation results from file if needed
        notify_validation_failed(["One or more validation checks failed"])
    elif event_type == "bias_failed":
        # Could parse bias details from file if needed
        notify_bias_failed({})
    elif event_type == "comparison_failed":
        # Could parse comparison details from file if needed
        notify_comparison_failed({
            'previous_score': 0,
            'current_score': 0,
            'improvement': 0,
            'improvement_pct': 0
        })
    else:
        print(f"Unknown event type: {event_type}")
        sys.exit(1)

