# mail_checker.py

from flask import Flask, jsonify
from mailer import send_email, init_mail
from dotenv import load_dotenv
import os

print("🚀 Starting Flask Mail Checker")

load_dotenv()

GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASS = os.getenv("GMAIL_APP_PASSWORD")

print(f"🔐 GMAIL_USER = {GMAIL_USER}")
print(f"🔐 GMAIL_APP_PASSWORD = {'SET' if GMAIL_PASS else 'MISSING'}")

app = Flask(__name__)

app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME=GMAIL_USER,
    MAIL_PASSWORD=GMAIL_PASS,
    MAIL_DEFAULT_SENDER=GMAIL_USER,
)

init_mail(app)

@app.route("/send-test", methods=["POST"])
def send_test():
    print("🛠️  /send-test route called")

    email = "wahidul.islam.ziad@gmail.com"
    subject = "📬 Flask Gmail SMTP Debug"
    body = """
    <h2>Hello Wahidul,</h2>
    <p>This is a debug test from Flask Gmail SMTP setup.</p>
    <p>If you received this, everything is working.</p>
    """

    success = send_email(email, subject, body)

    if success:
        print("🎉 Success response returned")
        return jsonify({"message": f"Email sent to {email}"}), 200
    else:
        print("💥 Failure response returned")
        return jsonify({"error": "Email sending failed"}), 500

if __name__ == "__main__":
    print("🧠 Flask app starting...")
    app.run(debug=True)
