[[ ! -d venv ]] && python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
export VOYAGE_API_KEY=$(cat VOYAGE_API_KEY)
export GEMINI_API_KEY=$(cat GEMINI_API_KEY)
python3 neighbors.py
deactivate
