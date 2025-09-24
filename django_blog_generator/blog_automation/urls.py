from django.urls import path
from . import views

app_name = 'blog_automation'

urlpatterns = [
    path('', views.home, name='home'),
    path('generate/<int:blog_id>/', views.generate_blog, name='generate_blog'),
    path('blog/<int:blog_id>/', views.blog_detail, name='blog_detail'),
    path('download/<int:blog_id>/', views.download_blog, name='download_blog'),
    path('list/', views.blog_list, name='blog_list'),
    path('status/<int:blog_id>/', views.check_status, name='check_status'),
    path('delete/<int:blog_id>/', views.delete_blog, name='delete_blog'),
]