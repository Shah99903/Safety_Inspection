# email_alert.py
import smtplib
import os
from dotenv import load_dotenv
from email.message import EmailMessage

load_dotenv()

EMAIL_ADDRESS = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 465))


def send_alert(subject, body, attachments=None, to=None):
    if to is None:
        to = EMAIL_RECEIVER

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = to
    msg.set_content(body)

    if attachments:
        for filepath in attachments:
            with open(filepath, "rb") as f:
                file_data = f.read()
                file_name = os.path.basename(filepath)
                msg.add_attachment(file_data, maintype="image", subtype="jpeg", filename=file_name)

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
            print("Email sent successfully.")
    except Exception as e:
        print("Failed to send email:", e)
