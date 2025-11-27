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
            
            # Import the parse_blog_titles function
            from .services import parse_blog_titles
            from django.db import transaction
            
            # Define maximum number of blogs to generate at once
            MAX_BLOGS_AT_ONCE = 5
            
            # Check if there are multiple titles (comma-separated)
            blog_titles, parsing_info = parse_blog_titles(blog_gen.title, max_titles=MAX_BLOGS_AT_ONCE)
            
            if len(blog_titles) > 1:
                # Multiple blog titles detected
                logger.info(f"Processing {len(blog_titles)} blog titles in batch")
                
                # Add parsing warnings to messages
                for warning in parsing_info.get('warnings', []):
                    messages.warning(request, f"Title processing: {warning}")
                
                # Generate multiple blogs (returns batch result)
                batch_result = generator.generate_multiple_blogs(
                    titles_input=blog_gen.title,
                    primary_keywords=blog_gen.primary_keywords,
                    num_competitors=blog_gen.num_competitors,
                    secondary_keywords=blog_gen.secondary_keywords,
                    blog_outline=blog_gen.blog_outline,
                    target_length=blog_gen.target_length,
                    max_titles=MAX_BLOGS_AT_ONCE
                )
                
                # Process batch results
                blog_results = batch_result.get('blog_results', [])
                successful = batch_result.get('successful_blogs', 0)
                failed = batch_result.get('failed_blogs', 0)
                batch_errors = batch_result.get('errors', [])
                generated_blogs = []
                
                # Use database transaction for efficiency and consistency
                with transaction.atomic():
                    # Process each result and create/update blog entries
                    for i, result in enumerate(blog_results):
                        blog_title = result.get('title', f"Blog {i+1}")
                        
                        if i == 0:
                            # Update the original blog entry with the first result
                            try:
                                if result['success']:
                                    content = result['content']
                                    blog_gen.formatted_post = content.get('formatted_post', '')
                                    blog_gen.word_count = content.get('word_count', 'N/A')
                                    blog_gen.competitor_analysis = content.get('comprehensive_analysis', content.get('content_strategy', {}))
                                    blog_gen.blog_analysis_data = content.get('blog_post_data', {})
                                    
                                    # Save as document
                                    doc_path = generator.save_as_document(
                                        formatted_post=content.get('formatted_post', ''),
                                        filename=blog_gen.filename
                                    )
                                    
                                    if doc_path:
                                        blog_gen.doc_file_path = doc_path
                                    
                                    blog_gen.status = 'completed'
                                    blog_gen.error_message = None
                                    blog_gen.title = blog_title  # Update with the exact title used
                                    generated_blogs.append(blog_gen)
                                else:
                                    blog_gen.status = 'failed'
                                    blog_gen.error_message = result.get('error', 'Unknown error occurred')
                                    blog_gen.title = blog_title
                            except Exception as e:
                                logger.error(f"Error updating original blog entry: {e}")
                                blog_gen.status = 'failed'
                                blog_gen.error_message = f"Error updating blog: {str(e)}"
                        else:
                            # Create new BlogGeneration objects for additional titles
                            try:
                                new_blog = BlogGeneration(
                                    title=blog_title,
                                    primary_keywords=blog_gen.primary_keywords,
                                    num_competitors=blog_gen.num_competitors,
                                    secondary_keywords=blog_gen.secondary_keywords,
                                    blog_outline=blog_gen.blog_outline,
                                    target_length=blog_gen.target_length
                                )
                                
                                if result['success']:
                                    content = result['content']
                                    new_blog.formatted_post = content.get('formatted_post', '')
                                    new_blog.word_count = content.get('word_count', 'N/A')
                                    new_blog.competitor_analysis = content.get('comprehensive_analysis', content.get('content_strategy', {}))
                                    new_blog.blog_analysis_data = content.get('blog_post_data', {})
                                    new_blog.status = 'completed'
                                    new_blog.save()
                                    
                                    # Save as document
                                    doc_path = generator.save_as_document(
                                        formatted_post=content.get('formatted_post', ''),
                                        filename=new_blog.filename
                                    )
                                    
                                    if doc_path:
                                        new_blog.doc_file_path = doc_path
                                        new_blog.save(update_fields=['doc_file_path'])
                                        
                                    generated_blogs.append(new_blog)
                                else:
                                    new_blog.status = 'failed'
                                    new_blog.error_message = result.get('error', 'Unknown error occurred')
                                    new_blog.save()
                                    
                            except Exception as e:
                                logger.error(f"Error creating new blog entry for '{blog_title}': {e}")
                
                # Set appropriate messages based on results
                if successful > 0 and failed == 0:
                    messages.success(request, f'🎉 Successfully generated all {successful} blogs!')
                elif successful > 0 and failed > 0:
                    messages.warning(request, f'⚠️ Generated {successful} blogs successfully, but {failed} failed.')
                    # Show specific errors for failed blogs
                    for error in batch_errors[-3:]:  # Show last 3 errors to avoid flooding
                        messages.error(request, error)
                    if len(batch_errors) > 3:
                        messages.error(request, f"... and {len(batch_errors) - 3} more errors. Check logs for details.")
                else:
                    messages.error(request, f'❌ Failed to generate any blogs.')
                    for error in batch_errors[:3]:  # Show first 3 errors
                        messages.error(request, error)
                
                # Save the updated original blog
                blog_gen.save()
                
                # Add batch summary to messages
                messages.info(request, f"Batch processing completed: {len(blog_titles)} titles processed, {successful} successful, {failed} failed")
                
                # If we have generated multiple blogs, redirect to the blog list page
                if len(generated_blogs) > 1:
                    messages.info(request, "Redirecting to blog list to view all generated blogs.")
                    return redirect('blog_automation:blog_list')
            
            else:
                # Single blog title - use the original approach
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
                    blog_gen.competitor_analysis = content.get('comprehensive_analysis', content.get('content_strategy', {}))
                    blog_gen.blog_analysis_data = content.get('blog_post_data', {})
                    
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