# CERAS React Migration — Setup Commands
# Run these commands in order from your terminal.

# ============================================
# STEP 1: Install Python backend dependencies
# ============================================
pip install fastapi uvicorn python-multipart

# ============================================
# STEP 2: Install React frontend dependencies
# ============================================
cd c:\Users\amanc\Desktop\ceras\frontend
npm install

# ============================================
# STEP 3: Start the FastAPI backend (Terminal 1)
# ============================================
# Open a NEW terminal and run:
cd c:\Users\amanc\Desktop\ceras
python server.py

# You should see:
#   ⏳ Loading ML models in background...
#   INFO:     Uvicorn running on http://0.0.0.0:8000
#   ✅ All ML models loaded successfully.

# ============================================
# STEP 4: Start the React frontend (Terminal 2)
# ============================================
# Open ANOTHER terminal and run:
cd c:\Users\amanc\Desktop\ceras\frontend
npm run dev

# You should see:
#   VITE v6.x.x  ready in xxx ms
#   ➜  Local:   http://localhost:5173/

# ============================================
# STEP 5: Open the app
# ============================================
# Visit http://localhost:5173 in your browser.
# The site will load immediately. 
# The sidebar will show "Loading ML models..." until they're ready.
# Once models are loaded, enter an API key and run a session!
