# 🚀 Railway Deployment Guide - SQLite Version

## ⚠️ Important Note About SQLite on Railway

- **Data will reset** on each deployment/restart
- **Generated files will be lost** during restarts
- **Perfect for demos** and small use cases
- **No persistent data storage**

## 📋 Prerequisites

- [ ] GitHub account with your repository
- [ ] Railway account (sign up at [railway.app](https://railway.app))
- [ ] API keys ready (SERP API, Google Gemini)

## 🔧 Step 1: Final Project Setup

Your project now has all the necessary files:

- ✅ `railway.json` - Railway configuration
- ✅ `Procfile` - Process definitions
- ✅ `runtime.txt` - Python version
- ✅ `requirements.txt` - Updated with gunicorn & whitenoise
- ✅ `settings_production.py` - Production Django settings

### Commit Your Changes

```bash
cd /Users/sanketpatel/Desktop/NetopDigital/Content_Generation_Automation
git add .
git commit -m "Add Railway deployment configuration"
git push origin main
```

## 🌐 Step 2: Deploy on Railway

### 1. Create Railway Account

- Go to [railway.app](https://railway.app)
- Sign up with GitHub (recommended)
- Verify your email

### 2. Create New Project

- Click **"New Project"**
- Select **"Deploy from GitHub repo"**
- Choose **"Configure GitHub App"** if first time
- Select your repository: **`Advance-SEO-Blog-Generation-App`**

### 3. Railway Auto-Detection

Railway will automatically:

- ✅ Detect it's a Python project
- ✅ Find your `railway.json` configuration
- ✅ Start the build process

## ⚙️ Step 3: Configure Environment Variables

In your Railway project dashboard:

### 1. Go to Variables Tab

- Click on your deployed service
- Navigate to **"Variables"** tab
- Add the following environment variables:

### 2. Required Variables

```bash
# Django Configuration
SECRET_KEY=your-super-secret-django-key-here
DEBUG=False
DJANGO_SETTINGS_MODULE=blog_generator.settings_production

# API Keys (Required for functionality)
SERP_API_KEY=your_actual_serp_api_key_from_serpapi_com
GOOGLE_GEMINI_API_KEY=your_actual_gemini_key_from_google

# Optional API Keys
DATAFORSEO_LOGIN=your_dataforseo_username
DATAFORSEO_PASSWORD=your_dataforseo_password
```

### 3. Generate Strong Secret Key

```bash
# Run this locally to generate a secret key:
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

## 🚀 Step 4: Deploy and Test

### 1. Trigger Deployment

- Railway will automatically deploy when you push to GitHub
- Or click **"Deploy"** in the Railway dashboard
- Monitor the build logs for any errors

### 2. Check Build Logs

Look for these success messages:

```
✅ Installing dependencies...
✅ Collecting static files...
✅ Running migrations...
✅ Starting gunicorn server...
```

### 3. Access Your App

- Railway will provide a URL like: `https://your-app-name.up.railway.app`
- Click the URL to access your deployed app
- Test the blog generation functionality

## 🔍 Step 5: Troubleshooting

### Common Issues and Solutions:

#### 1. **Build Fails - Dependencies**

```bash
# Error: No module named 'xyz'
# Solution: Check requirements.txt has all packages
```

#### 2. **Static Files Not Loading**

```bash
# Error: CSS/JS not working
# Check: STATIC_ROOT and whitenoise configuration
# Our settings_production.py handles this
```

#### 3. **Environment Variables Not Working**

```bash
# Error: API keys not found
# Check: Variables tab in Railway dashboard
# Ensure no extra spaces in variable values
```

#### 4. **Database Errors**

```bash
# Error: Database tables don't exist
# Solution: Railway runs migrations automatically
# Check build logs for migration success
```

## 📊 Railway Dashboard Features

### 1. **Monitoring**

- View real-time logs
- Monitor resource usage
- Check deployment history

### 2. **Logs**

```bash
# View application logs:
# - Build logs
# - Runtime logs
# - Error logs
```

### 3. **Custom Domain (Optional)**

- Go to **"Settings"** → **"Domains"**
- Add your custom domain
- Railway provides free SSL

## 🔄 Ongoing Operations

### 1. **Automatic Deployments**

- Push to GitHub → Railway auto-deploys
- No manual intervention needed
- Build logs show deployment progress

### 2. **Database Considerations**

```bash
# Remember: SQLite resets on each deployment
# User data will be lost
# Generated blogs will be lost
# Perfect for demos and testing
```

### 3. **Updating Your App**

```bash
# Local development:
git add .
git commit -m "Update feature"
git push origin main

# Railway will automatically deploy the changes
```

## 💡 Tips for Success

### 1. **API Key Management**

- Keep API keys secure
- Never commit them to git
- Use Railway's environment variables

### 2. **Monitoring Usage**

```bash
# Railway free tier includes:
# - 500 execution hours/month
# - $5 credit monthly
# - Automatic sleep after inactivity
```

### 3. **Performance Optimization**

- SQLite is fast for small datasets
- Consider caching for API calls
- Monitor Railway resource usage

## 🎯 Testing Checklist

After deployment, test these features:

- [ ] Home page loads correctly
- [ ] Blog generation form works
- [ ] API keys are recognized
- [ ] Blog generation completes
- [ ] Document download works
- [ ] Static files (CSS/JS) load properly

## 🆘 Support Resources

### Railway Support

- [Railway Documentation](https://docs.railway.app)
- [Railway Community Discord](https://discord.gg/railway)
- [Railway GitHub Issues](https://github.com/railwayapp/railway/issues)

### Project Support

- Check build logs in Railway dashboard
- Review Django logs for application errors
- Verify environment variables are set correctly

---

## 🎉 Congratulations!

Your Django SEO Blog Generator is now live on Railway!

**Your app URL:** `https://your-app-name.up.railway.app`

**Remember:** This SQLite deployment is perfect for:

- ✅ Demos and presentations
- ✅ Testing and development
- ✅ Small-scale usage
- ✅ Proof of concept

For production with persistent data, consider upgrading to PostgreSQL on Railway.
