from flask_mail import Mail, Message
import threading

mail = Mail()
_app = None  # Global reference to Flask app

def init_mail(app):
    global _app
    _app = app
    print("🔧 Initializing Flask-Mail with app config")
    mail.init_app(app)

def send_email(to, subject, html):
    try:
        msg = Message(subject=subject, recipients=[to], html=html)
        mail.send(msg)
        print("✅ Email sent successfully.")
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False

def send_email_async(to, subject, html):
    def send_with_context():
        with _app.app_context():
            send_email(to, subject, html)
    threading.Thread(target=send_with_context).start()