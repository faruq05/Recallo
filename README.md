# 📚 Recallo: AI-Driven Spaced Repetition & Recall Booster for Academic Success
Recallo is an intelligent platform that leverages AI and the science of spaced repetition to help students retain and recall information effectively. This monorepo contains the complete frontend, backend, and AI engine setup.

## 🚀 Quick Start Guide

1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Recallo.git
cd Recallo
```

2️⃣ Run the Setup Script

```python
npm run setup
```

## 🧠 VS Code Python Interpreter Setup

If you're using VS Code and Python extension must be installed, do the following to activate the virtual environment:

1. Press Ctrl + Shift + P to open Command Palette

2. Type: Python: Select Interpreter

3. Choose: Enter Interpreter Path

4. Click Find...

5. Navigate to:

```python
Recallo/ai-engine/.venv/Scripts/python.exe
```
6. Click Select Interpreter
7. ✅ You’re all set! VS Code will now use the correct environment.


## 🔐 .env Configuration

After npm run setup, a .env file will be created inside the backend/ directory.
You must replace the placeholders with your actual credentials:
```python
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
GEMINI_API_KEY=your_gemini_api_key
```

## 💻 How to Run the Application
1. Start the frontend
```python
cd frontend
npm run dev
```
2. Activate the Virtual Environment (for AI Engine)
```python
cd ai-engine
.venv\Scripts\activate  # Windows
# Or
source .venv/bin/activate  # macOS/Linux
```
3.  Run the Backend Server
```python
cd backend
python app.py
```

## ✅ Everything Ready?
Your Recallo AI platform should now be running and accessible locally.
Use it to boost memory retention, improve academic performance, and learn smarter — not harder.
