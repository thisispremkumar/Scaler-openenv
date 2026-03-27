# 🔐 HuggingFace Authentication Required

## Your Situation

You're trying to push to HF Spaces but getting:
```
error: failed to push some refs
You are not authorized to push to this repo.
```

**Solution**: You need to provide your HuggingFace API token for authentication.

---

## Quick Fix (2 Steps)

### Step 1: Get Your HF Token

Go to: **https://huggingface.co/settings/tokens**

1. Click **"New token"**
2. Name: `Scaler Deployment`
3. Click **"Generate"**
4. **Copy the token** (save it - it won't show again!)
5. It should start with `hf_`

### Step 2: Push with Token

Run this in PowerShell (replace `YOUR_HF_TOKEN` with your actual token):

```powershell
cd C:\Users\premm\Desktop\Projects\Openenv\Scaler-openenv
git push https://premmokara:YOUR_HF_TOKEN@huggingface.co/spaces/premmokara/Scaler-openenv.git main
```

**Example** (with fake token):
```powershell
git push https://premmokara:hf_1234567890abcdefghijklmnop@huggingface.co/spaces/premmokara/Scaler-openenv.git main
```

---

## What to Replace

- `YOUR_HF_TOKEN` = Your actual token from https://huggingface.co/settings/tokens
- `premmokara` = Your HuggingFace username (should be correct already)
- Everything else stays the same

---

## Expected Output

If successful, you'll see:
```
Counting objects: 25, done.
Writing objects: 100% (25/25), done.
Total 23 (delta 3), reused 0 (delta 0), pack-reused 0 (from 0)
remote: Synced commiters with the Hub.
To https://huggingface.co/spaces/premmokara/Scaler-openenv
 * [new branch]      main -> main
```

Then HF Spaces will automatically:
1. Detect the Dockerfile
2. Build Docker image (2-5 minutes)
3. Deploy your environment
4. Start the container on port 7860

---

## ⚠️ Security Notes

- Your token acts like a password - keep it secure
- Don't paste it in chat or public places
- It only appears once - save it safely
- Can be revoked anytime from HF settings

---

## Still Having Issues?

**If push still fails:**
1. Verify you copied the FULL token (starts with `hf_`)
2. Check your username is correct (`premmokara`)
3. Make sure the Space URL is correct
4. Try in a fresh PowerShell window (in case of caching)

**Check if this helps:**
```powershell
# Test HF token
curl https://huggingface.co/api/user -H "Authorization: Bearer YOUR_HF_TOKEN"
```

Should return your profile info (not an error).

---

## Once Pushed Successfully

After the push completes:
1. Go to: https://huggingface.co/spaces/premmokara/Scaler-openenv
2. Watch the build (might show "Building..." for 2-5 minutes)
3. Once "Running", add secrets:
   - Go to Settings → Repository secrets
   - Add `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
4. Test: https://huggingface.co/spaces/premmokara/Scaler-openenv/health

---

**Ready?** Get your HF token and run the push command above! 🚀
