from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse, FileResponse, Http404
from django.views.decorators.http import require_http_methods
from django.urls import reverse
from django.utils import timezone
import json
import os
import logging

from .models import BlogGeneration
from .forms import BlogGenerationForm
from .services import SEOBlogGenerator, check_api_configuration

logger = logging.getLogger(__name__)


def home(request):
    """Home page with blog generation form"""
    # Check API configuration
    api_status = check_api_configuration()
    
    if request.method == 'POST':
        form = BlogGenerationForm(request.POST)
        if form.is_valid():
            # Save the blog generation request
            blog_gen = form.save()
            blog_gen.status = 'pending'
            blog_gen.save()
            
            # Redirect to processing page
            return redirect('blog_automation:generate_blog', blog_id=blog_gen.id)
    else:
        form = BlogGenerationForm()
    
    # Get recent blog generations
    recent_blogs = BlogGeneration.objects.all()[:10]
    
    context = {
        'form': form,
        'recent_blogs': recent_blogs,
        'api_status': api_status,
    }
    
    return render(request, 'blog_automation/home.html', context)


def generate_blog(request, blog_id):
    """Process blog generation for a specific request"""
    blog_gen = get_object_or_404(BlogGeneration, id=blog_id)
    
    if request.method == 'POST':
        # Start the blog generation process
        blog_gen.status = 'processing'
        blog_gen.updated_at = timezone.now()
        blog_gen.save()
        
        try:
            # Initialize the SEO blog generator
            generator = SEOBlogGenerator()
            
            # Generate the blog
            result = generator.generate_blog(
                title=blog_gen.title,
                primary_keywords=blog_gen.primary_keywords,
                num_competitors=blog_gen.num_competitors,
                secondary_keywords=blog_gen.secondary_keywords,
                blog_outline=blog_gen.blog_outline,
                target_length=blog_gen.target_length
            )
            
            if result['success']:
                content = result['content']
                
                # Save the results to the database
                blog_gen.formatted_post = content.get('formatted_post', '')
                blog_gen.word_count = content.get('word_count', 'N/A')
                blog_gen.competitor_analysis = content.get('content_strategy', {})
                blog_gen.blog_outline = content.get('blog_post_data', {})
                
                # Save as document
                doc_path = generator.save_as_document(
                    formatted_post=content.get('formatted_post', ''),
                    filename=blog_gen.filename
                )
                
                if doc_path:
                    blog_gen.doc_file_path = doc_path
                
                blog_gen.status = 'completed'
                blog_gen.error_message = None
                
                messages.success(request, 'Blog generated successfully!')
                
            else:
                blog_gen.status = 'failed'
                blog_gen.error_message = result.get('error', 'Unknown error occurred')
                messages.error(request, f"Blog generation failed: {result.get('error')}")
            
        except Exception as e:
            logger.error(f"Error generating blog {blog_id}: {str(e)}")
            blog_gen.status = 'failed'
            blog_gen.error_message = str(e)
            messages.error(request, f"An error occurred: {str(e)}")
        
        finally:
            blog_gen.updated_at = timezone.now()
            blog_gen.save()
        
        return redirect('blog_automation:blog_detail', blog_id=blog_gen.id)
    
    context = {
        'blog_gen': blog_gen,
    }
    
    return render(request, 'blog_automation/generate.html', context)


def blog_detail(request, blog_id):
    """Display blog generation details and results"""
    blog_gen = get_object_or_404(BlogGeneration, id=blog_id)
    
    context = {
        'blog_gen': blog_gen,
    }
    
    return render(request, 'blog_automation/detail.html', context)


def download_blog(request, blog_id):
    """Download the generated blog document"""
    blog_gen = get_object_or_404(BlogGeneration, id=blog_id)
    
    if not blog_gen.doc_file_path or not os.path.exists(blog_gen.doc_file_path):
        raise Http404("Document not found")
    
    try:
        response = FileResponse(
            open(blog_gen.doc_file_path, 'rb'),
            content_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )
        response['Content-Disposition'] = f'attachment; filename="{blog_gen.filename}"'
        return response
    except Exception as e:
        logger.error(f"Error downloading file {blog_gen.doc_file_path}: {str(e)}")
        raise Http404("Error downloading document")


def blog_list(request):
    """List all blog generations"""
    blogs = BlogGeneration.objects.all()
    
    context = {
        'blogs': blogs,
    }
    
    return render(request, 'blog_automation/list.html', context)


@require_http_methods(["GET"])
def check_status(request, blog_id):
    """AJAX endpoint to check blog generation status"""
    blog_gen = get_object_or_404(BlogGeneration, id=blog_id)
    
    data = {
        'status': blog_gen.status,
        'updated_at': blog_gen.updated_at.isoformat(),
        'error_message': blog_gen.error_message,
        'word_count': blog_gen.word_count,
    }
    
    return JsonResponse(data)


def delete_blog(request, blog_id):
    """Delete a blog generation request"""
    blog_gen = get_object_or_404(BlogGeneration, id=blog_id)
    
    if request.method == 'POST':
        # Delete associated file if it exists
        if blog_gen.doc_file_path and os.path.exists(blog_gen.doc_file_path):
            try:
                os.remove(blog_gen.doc_file_path)
            except Exception as e:
                logger.warning(f"Could not delete file {blog_gen.doc_file_path}: {e}")
        
        blog_gen.delete()
        messages.success(request, 'Blog deleted successfully.')
        return redirect('blog_automation:home')
    
    context = {
        'blog_gen': blog_gen,
    }
    
    return render(request, 'blog_automation/delete_confirm.html', context)