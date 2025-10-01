"""
Production settings for Railway deployment with SQLite
"""
import os
from .settings import *

# Production mode
DEBUG = False

# Railway provides PORT environment variable
PORT = int(os.environ.get('PORT', 8000))

# Platform-specific allowed hosts
ALLOWED_HOSTS = [
    'localhost',
    '127.0.0.1',
    '.railway.app',      # Railway
    '.up.railway.app',   # Railway
    '.onrender.com',     # Render
    '.render.com',       # Render
    '.pythonanywhere.com', # PythonAnywhere
    '.fly.dev',          # Fly.io
]

# Add custom domain if provided
if 'RAILWAY_PUBLIC_DOMAIN' in os.environ:
    ALLOWED_HOSTS.append(os.environ.get('RAILWAY_PUBLIC_DOMAIN'))
if 'RENDER_EXTERNAL_HOSTNAME' in os.environ:
    ALLOWED_HOSTS.append(os.environ.get('RENDER_EXTERNAL_HOSTNAME'))

# Use environment variable for secret key
SECRET_KEY = os.environ.get('SECRET_KEY', 'django-insecure-fallback-key-change-in-production')

# Database - SQLite (will reset on each deployment)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Static files configuration for Railway
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Add whitenoise to middleware (already done below)
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',  # Add this for serving static files
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# Configure whitenoise
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'
WHITENOISE_USE_FINDERS = True
WHITENOISE_AUTOREFRESH = True

# Media files (ephemeral on Railway)
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Blog generation settings - optimize for memory in production
OPTIMIZE_FOR_LOW_MEMORY = True  # Enable memory optimization for Render/Railway

# Security settings (relaxed for SQLite deployment)
SECURE_SSL_REDIRECT = False
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE = False

# CORS settings (if needed)
CORS_ALLOW_ALL_ORIGINS = True  # Only for development/demo

# Logging configuration for Railway
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
        'blog_automation': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}