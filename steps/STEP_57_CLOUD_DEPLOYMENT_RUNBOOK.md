# Step 57: Cloud Deployment Runbook (Hugging Face / Render / Fly.io)

## 1. Process & Purpose
Deploy AniRec to a live, publicly accessible cloud URL on free-tier or low-cost infrastructure (Hugging Face Spaces, Render Web Service, or Fly.io).

---

## 2. Platform Deployment Matrix

| Platform | Best For | Configuration Method | Free Tier / Cost |
|---|---|---|---|
| **Hugging Face Spaces** | Direct Streamlit app deployment | Docker space or Streamlit SDK space | Free 2 vCPU / 16GB RAM |
| **Render** | Standalone FastAPI web service + Streamlit | Web Service connected to GitHub Repo | Free tier available |
| **Fly.io** | Production global container deployment | `fly launch` + `fly.toml` | Low cost / generous initial credits |

---

## 3. Step-by-Step Deployment Guide (Hugging Face Spaces)

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces) with Space SDK: **Docker**.
2. Set Space Secret: `OPENAI_API_KEY = sk-...`.
3. In `README.md` at repo root, add the HF Space metadata header:
   ```yaml
   ---
   title: AniRec - AI Anime Recommendation Engine
   emoji: 🎌
   colorFrom: purple
   colorTo: indigo
   sdk: docker
   app_port: 8501
   ---
   ```
4. Push repo to Hugging Face Git remote:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/anirec
   git push space main
   ```

---

## 4. Deliverables & Acceptance Criteria
- [x] Application successfully deployed to a public URL.
- [x] Live link tested and documented for recruiter review.
