#!/bin/bash

# Navigate to Django directory
cd django_blog_generator

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput

# Run migrations  
echo "Running migrations..."
python manage.py migrate

# Start gunicorn server
echo "Starting gunicorn server on 0.0.0.0:$PORT..."
exec gunicorn blog_generator.wsgi:application --bind 0.0.0.0:$PORT --workers 2 --timeout 120