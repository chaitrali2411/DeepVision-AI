# Hugging Face login for Gemma 3n

## 1. Get a valid token (not your password)

1. Open https://huggingface.co/settings/tokens
2. Sign in to Hugging Face.
3. Click **"Create new token"**.
4. Name it (e.g. `roomcare`), set **Role: Read**.
5. Click **Create** and **copy the token**. It must start with `hf_` (e.g. `hf_xxxxxxxxxxxx`).
   - Use the **token**, not your account password.

## 2. Accept Gemma 3n terms

1. Open https://huggingface.co/google/gemma-3n-e2b-it
2. If you see **"Agree and access repository"**, click it and accept.

## 3. Log in using the same Python you use for training

Your training uses **pyenv Python 3.10**. Run login with that Python so the token is stored for it:

```bash
cd /Users/chaitralikadam/Documents/DeepVision_AI
# Use pyenv Python (same as training)
python -m pip install -U huggingface_hub
python -c "from huggingface_hub import login; login()"
```

When prompted:
- Paste your **token** (the one starting with `hf_`). Nothing will show as you paste — that’s normal.
- Press **Enter**.
- You should see a success message.

Or in one shot (replace `YOUR_TOKEN` with your real token):

```bash
python -c "from huggingface_hub import login; login(token='YOUR_TOKEN')"
```

## 4. Run training again

```bash
python -m roomcare.train --data_dir roomcare/dataset --output_dir roomcare/checkpoints/roomcare_lora
```

## If it still says "Invalid user token"

- Create a **new** token at https://huggingface.co/settings/tokens and try again.
- Make sure you copied the full token (starts with `hf_`, no spaces).
- Do not use your Hugging Face account password; only the token works.
