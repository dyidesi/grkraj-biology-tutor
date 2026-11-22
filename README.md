# 🧬 grkraj.org Biology Study Tutor (Streamlit Dark-Mode Web App + Offline Gradio UI)

Interactive Molecular Biology, Plant Cell Biology, and Pre-University Biology tutor built over all **222 chapters (`6,317` cleaned lecture passages + `9,234` biology diagrams)** of **Prof. Dr. G. R. Kantharaj** (Bangalore University, https://grkraj.org).

---

## Option 1 — Deploy FREE Forever on Streamlit Community Cloud (`app.py`)

`app.py` is a lightweight **Bioluminescent Obsidian dark-mode Streamlit app** (~250 MB RAM, zero GPU needed) powered by the public **`sgowda/grkraj`** `gemini-embedding-2` index (`67 MB`, auto-downloaded at startup with zero HF token required) + **`gemini-3.8-flash`**.

### Deploy in 3 Steps (\$0 Hosting!)
1. Push this `student-app/` folder (`app.py`, `requirements.txt`, `.streamlit/config.toml`) to any public **GitHub** repository.
2. Go to **https://share.streamlit.io** → click **Create app** → select your repo and set **Main file path** to `app.py` (or `student-app/app.py`).
3. Click **Advanced settings → Secrets** and add your Gemini API key:
   ```toml
   GEMINI_API_KEY = "AIza..."
   DAILY_FREE_LIMIT = "20"
   GLOBAL_DAILY_LIMIT = "100"
   ```
   *(Optional: you can also deploy with **no server key** — the prominent Gemini API Key card guides students to paste their own free key in 30 seconds!)*
4. Click **Deploy** — share the permanent `https://<your-app>.streamlit.app` link with students! Works on any phone, Chromebook, or 8 GB laptop.

### Run `app.py` Locally
```powershell
cd student-app
pip install -r requirements.txt
$env:GEMINI_API_KEY = "AIza..."
streamlit run app.py
# Open http://localhost:8501
```

---

## Option 2 — 100% Offline Dark-Mode Local Web UI (`app_offline.py`, Zero API Keys)

Runs completely offline using **`Qwen/Qwen3-VL-Embedding-2B`** + **`Qwen/Qwen3-VL-8B-Instruct`** (24 GB NVIDIA GPU) or **LM Studio** (16 GB laptop):
```powershell
cd student-app
python app_offline.py
# Open http://127.0.0.1:7861
```
*(For beginners, see [`../STUDENT_QUICKSTART.md`](../STUDENT_QUICKSTART.md) and double-click `Start_Biology_Tutor.bat` or `Start_Biology_Tutor.command`.)*
