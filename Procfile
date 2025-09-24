web: cd django_blog_generator && gunicorn blog_generator.wsgi:application --bind 0.0.0.0:$PORT
release: cd django_blog_generator && python manage.py migrate && python manage.py collectstatic --noinput