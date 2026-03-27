# 🚀 Complete HF Spaces Deployment Guide

## Status
✅ Project cloned to: `Scaler-openenv`  
✅ Files copied and committed  
⏳ **Pending**: Push to HF Spaces (needs your HF_TOKEN)

---

## Step 1: Get Your HF_TOKEN

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it (e.g., "Scaler Deployment")
4. Click "Generate"
5. **Copy the token** (starts with `hf_`)

---

## Step 2: Configure Git with HF_TOKEN

In PowerShell, run these commands (replace `YOUR_TOKEN` with your actual token):

```powershell
# Set git credential helper to store credentials
git config --global credential.helper store

# Navigate to Scaler-openenv
cd C:\Users\premm\Desktop\Projects\Openenv\Scaler-openenv

# Try pushing (will prompt for credentials)
git push origin main
```

When prompted:
- **Username**: `premmokara` (your HF username)
- **Password**: `YOUR_TOKEN` (paste your HF token here)

Git will remember these credentials for future pushes.

---

## Alternative Step 2: Use HTTPS with Token in URL

Instead, you can push with token embedded in the URL:

```powershell
cd C:\Users\premm\Desktop\Projects\Openenv\Scaler-openenv

# Replace YOUR_TOKEN with your actual token
git push https://premmokara:YOUR_TOKEN@huggingface.co/spaces/premmokara/Scaler-openenv.git main
```

---

## Step 3: Wait for HF Space to Build

Once pushed, HuggingFace will automatically:
1. Detect the Dockerfile
2. Build the Docker image
3. Deploy to your Space
4. Run the container on port 7860

This takes 2-5 minutes.

---

## Step 4: Add Secrets to Your HF Space

After the Space is running:

1. Go to: https://huggingface.co/spaces/premmokara/Scaler-openenv
2. Click **Settings** (gear icon, top right)
3. Scroll to **Repository secrets**
4. Click **New secret** for each:
   - **Name**: `API_BASE_URL`  
     **Value**: `https://router.huggingface.co/v1`
   - **Name**: `MODEL_NAME`  
     **Value**: `nvidia/llama-3.1-nemotron-70b-instruct`
   - **Name**: `HF_TOKEN`  
     **Value**: `hf_YOUR_TOKEN_HERE` (paste your token)

5. Space will restart automatically

---

## Step 5: Update Local .env File

In `Scaler-openenv\.env`, update:

```
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=nvidia/llama-3.1-nemotron-70b-instruct
HF_TOKEN=hf_YOUR_TOKEN_HERE
HF_SPACE_URL=https://huggingface.co/spaces/premmokara/Scaler-openenv
```

---

## Step 6: Test Your Deployment

Once Space is running, test the endpoint:

```bash
# Test health check
curl https://huggingface.co/spaces/premmokara/Scaler-openenv/health
```

Should return: `{"status":"healthy"}`

---

## Step 7: Run Validator

```bash
cd Scaler-openenv
python -m validate_submission
```

Expected output: `PASS_PENDING_HUMAN_REVIEW`

---

## What Was Already Done

✅ Cloned HF Space repository  
✅ Copied all project files (models, server, tasks, tests)  
✅ Created `.env` template  
✅ Updated README  
✅ Committed all changes  
✅ Ready to push (just needs your credentials)

---

## Files in Scaler-openenv

```
Scaler-openenv/
├── __init__.py
├── baseline_inference.py
├── client.py
├── inference.py
├── judging.py
├── models.py
├── validate_submission.py
├── openenv.yaml
├── pyproject.toml
├── .env                 ← Add your credentials here
├── .gitignore
├── README.md
├── server/
│   ├── app.py
│   ├── Dockerfile
│   ├── my_real_world_env_environment.py
│   ├── requirements.txt
│   ├── tasks.py
│   └── __init__.py
└── tests/
    ├── test_environment.py
    └── test_graders.py
```

---

## Next: Push Your Code

**Copy and run this in PowerShell** (replace YOUR_TOKEN):

```powershell
cd C:\Users\premm\Desktop\Projects\Openenv\Scaler-openenv
git push https://premmokara:YOUR_TOKEN@huggingface.co/spaces/premmokara/Scaler-openenv.git main
```

Once completed, your HF Space will auto-build and deploy! 🚀
