#!/bin/bash

# Railway startup script
echo "🚀 Starting Django app deployment..."

# Navigate to Django directory
cd django_blog_generator

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r ../requirements.txt

# Run migrations
echo "🗄️ Running database migrations..."
python manage.py migrate

# Collect static files
echo "📁 Collecting static files..."
python manage.py collectstatic --noinput

# Start the server
echo "🌐 Starting Gunicorn server..."
exec gunicorn blog_generator.wsgi:application --bind 0.0.0.0:$PORT