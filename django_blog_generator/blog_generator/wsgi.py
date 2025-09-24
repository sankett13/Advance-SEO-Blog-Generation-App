"""
WSGI config for blog_generator project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

# Use production settings on Railway, development settings locally
if 'RAILWAY_ENVIRONMENT' in os.environ or 'PORT' in os.environ:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'blog_generator.settings_production')
else:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'blog_generator.settings')

application = get_wsgi_application()
