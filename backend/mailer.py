from flask_mail import Mail, Message

mail = Mail()  # Will be initialized from app

def init_mail(app):
    print("🔧 Initializing Flask-Mail with app config")
    mail.init_app(app)

def send_email(to, subject, html):
    print("📤 Preparing to send email...")
    print(f"📧 To: {to}")
    print(f"📨 Subject: {subject}")
    print(f"💬 Body:\n{html}")

    try:
        msg = Message(subject=subject, recipients=[to], html=html)
        print("✉️ Message object created.")
        mail.send(msg)
        print("✅ Email sent successfully.")
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False
