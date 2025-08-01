import os
import logging
from flask_mail import Mail, Message
from flask import Flask
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create a minimal Flask app for email sending
email_app = Flask(__name__)
email_app.config['MAIL_SERVER'] = 'smtp.gmail.com'
email_app.config['MAIL_PORT'] = 587
email_app.config['MAIL_USE_TLS'] = True
email_app.config['MAIL_USERNAME'] = os.environ.get("MAIL_USERNAME")
email_app.config['MAIL_PASSWORD'] = os.environ.get("MAIL_PASSWORD")
email_app.config['MAIL_DEFAULT_SENDER'] = os.environ.get("MAIL_USERNAME")

# Initialize Mail with the email app
mail = Mail(email_app)

def send_email(to, subject, body):
    """Use Flask-Mail to send the message."""
    try:
        with email_app.app_context():
            msg = Message(subject, recipients=[to])
            msg.body = body
            mail.send(msg)
            return True
    except Exception as e:
        logging.error(f"Mail send failed: {e}")
        return False

def send_exam_result_email(user_email, user_name, topic_title, score):
    """Send a detailed and friendly result email after quiz submission."""
    subject = f"🎯 Your Quiz Result: {topic_title}"

    if score >= 8:
        message = f"""Hi {user_name},

🎉 Congratulations on your excellent performance!

You scored {score:.1f}/10 in the topic: "{topic_title}".

Keep up the great work and continue sharpening your skills. You're doing fantastic!

Best wishes,  
The Recallo Team

---
📚 Ready for more challenges? Visit your dashboard to explore new topics!
🔗 https://recallo.ai/dashboard
"""
    elif score >= 5:
        message = f"""Hi {user_name},

👍 You scored {score:.1f}/10 on the topic: "{topic_title}".

That's a solid effort! With a little more practice, you'll master this topic in no time. Would you like to retake it for a better score?

💡 Tip: Review the explanations and try again to boost your score to 8+ and stop daily reminders!

Stay motivated!  
The Recallo Team

---
🔄 Retake Quiz: https://recallo.ai/exam
📖 Study Materials: https://recallo.ai/topics
"""
    else:
        message = f"""Hi {user_name},

💡 You scored {score:.1f}/10 on the topic: "{topic_title}".

Don't be discouraged! Every expert was once a beginner. This is your chance to come back stronger — give it another go and improve your score.

🎯 Goal: Aim for 8+ to master this topic and stop daily reminders!

We're rooting for you!  
The Recallo Team

---
🔄 Try Again: https://recallo.ai/exam
📚 Study More: https://recallo.ai/topics
💬 Need Help?: https://recallo.ai/chat
"""

    return send_email(user_email, subject, message.strip())

def send_reminder_email(user_email, user_name, topic_title, score, reminder_type="daily"):
    """Send reminder email for practice."""
    if reminder_type == "daily":
        subject = f"🎯 Daily Reminder: Improve your score on {topic_title}"
        message = f"""Hi {user_name},

📚 Time to boost your knowledge!

You scored {score:.1f}/10 on "{topic_title}". Ready to aim for 8+ and master this topic?

🎯 Take the quiz again and stop these daily reminders by scoring 8 or higher!

💪 You've got this!

Best regards,
The Recallo Team

---
🔄 Retake Quiz: https://recallo.ai/exam
⚙️ Manage Notifications: https://recallo.ai/settings
"""
    else:  # weekly
        subject = f"🌟 Weekly Practice: Keep your knowledge fresh on {topic_title}"
        message = f"""Hi {user_name},

🧠 Time for your weekly practice session!

Great job on scoring {score:.1f}/10 on "{topic_title}"! Keep your knowledge sharp with a quick review.

📖 A little practice goes a long way in retaining what you've learned.

Keep up the excellent work!

Best regards,
The Recallo Team

---
🔄 Practice Again: https://recallo.ai/exam
📚 Explore Topics: https://recallo.ai/topics
"""

    return send_email(user_email, subject, message.strip())

# Test function
def test_email_sending():
    """Test email sending functionality"""
    test_email = "test@example.com"
    test_name = "Test User"
    test_topic = "Sample Topic"
    test_score = 7.5
    
    try:
        result = send_exam_result_email(test_email, test_name, test_topic, test_score)
        print(f"Test email result: {result}")
        return result
    except Exception as e:
        print(f"Test email failed: {e}")
        return False

if __name__ == "__main__":
    # Test the email functionality
    test_email_sending()
