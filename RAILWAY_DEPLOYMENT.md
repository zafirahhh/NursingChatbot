# Railway Deployment Guide

## Step 1: Prepare for Railway Deployment

✅ Files are ready:
- `railway.json` - Railway configuration
- `requirements.txt` - Python dependencies
- `backend.py` - Updated to use PORT environment variable

## Step 2: Deploy to Railway

1. **Go to [Railway.app](https://railway.app)**
2. **Sign in with GitHub**
3. **Click "New Project"**
4. **Select "Deploy from GitHub repo"**
5. **Choose your repository**
6. **Railway will automatically:**
   - Detect it's a Python project
   - Install dependencies from `requirements.txt`
   - Start the server using `railway.json` configuration

## Step 3: Update Frontend URLs

After deployment, Railway will give you a URL like:
`https://your-app-name.up.railway.app`

**Update `js/config.js`:**
```javascript
const BACKEND_URL_PROD = "https://YOUR-ACTUAL-URL.up.railway.app/ask";
const QUIZ_URL_PROD = "https://YOUR-ACTUAL-URL.up.railway.app/quiz";
const QUIZ_EVAL_URL_PROD = "https://YOUR-ACTUAL-URL.up.railway.app/quiz/evaluate";
```

## Step 4: Test Your Deployment

1. **Check Railway logs** to ensure no errors
2. **Test the API endpoints:**
   - `https://your-url.up.railway.app/quiz` (should return JSON)
3. **Open your GitHub Pages site** - it should now work!

## Step 5: GitHub Pages Setup (Optional)

If you want the frontend on GitHub Pages:

1. Go to your GitHub repo → Settings → Pages
2. Source: Deploy from branch → `main` → `/` (root)
3. Your site will be at: `https://username.github.io/repository-name`

## Troubleshooting

**If deployment fails:**
- Check Railway logs in the dashboard
- Ensure `requirements.txt` has all dependencies
- Verify the `PORT` environment variable is being used

**If frontend can't connect:**
- Check browser console for CORS errors
- Verify the URLs in `config.js` match your Railway deployment URL
- Test API endpoints directly in browser

## Environment Variables (if needed)

In Railway dashboard, you can set:
- `PYTHON_VERSION` = `3.9`
- Any other config your app needs
