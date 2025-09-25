"""
SEO Content Automation Service
Integrates the existing seo_content_automation.py functionality into Django
"""
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Add the parent directory to sys.path to import the original script
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Import functions from the original script
try:
    import seo_content_automation
    from seo_content_automation import (
        create_complete_blog_workflow,
        format_blog_post_for_publication,
        save_as_docx
    )
    logging.info(f"Successfully imported SEO automation functions from {parent_dir}")
except ImportError as e:
    logging.error(f"Could not import SEO automation functions: {e}")
    logging.error(f"Tried to import from: {parent_dir}")
    logging.error(f"Files in parent dir: {list(parent_dir.glob('*.py'))}")
    # Fallback imports if needed
    seo_content_automation = None
    create_complete_blog_workflow = None
    format_blog_post_for_publication = None
    save_as_docx = None

from django.conf import settings
from django.core.files.base import ContentFile
import tempfile

logger = logging.getLogger(__name__)


class SEOBlogGenerator:
    """Service class to handle SEO blog generation using the existing automation script"""
    
    def __init__(self):
        self.media_root = getattr(settings, 'MEDIA_ROOT', tempfile.gettempdir())
        
    def generate_blog(self, title, primary_keywords, num_competitors=3, secondary_keywords=None, blog_outline=None, target_length=None):
        """
        Generate a complete blog post using the existing SEO automation script
        
        Args:
            title (str): Blog title or target keyword
            primary_keywords (str): Comma-separated primary keywords
            num_competitors (int): Number of competitors to analyze
            secondary_keywords (str): Optional secondary keywords
            blog_outline (str): Optional custom blog outline
            target_length (str): Optional target blog length
            
        Returns:
            dict: Result containing blog content and metadata
        """
        try:
            # Check for required environment variables
            if not os.getenv('GOOGLE_GEMINI_API_KEY'):
                return {
                    'success': False,
                    'error': 'GOOGLE_GEMINI_API_KEY environment variable is not set. Please configure it in Render dashboard.',
                    'content': None
                }
                
            if not os.getenv('SERP_API_KEY'):
                return {
                    'success': False,
                    'error': 'SERP_API_KEY environment variable is not set. Please configure it in Render dashboard.',
                    'content': None
                }
            
            # Prepare the target keyword (use title as primary target)
            target_keyword = title
            
            logger.info(f"Starting blog generation for: {target_keyword}")
            logger.info(f"Optional parameters - Secondary keywords: {secondary_keywords}, Outline: {blog_outline}, Length: {target_length}")
            
            # Check if the original functions are available
            if not create_complete_blog_workflow:
                return {
                    'success': False,
                    'error': 'SEO automation functions not available. Please check imports.',
                    'content': None
                }
            
            # Run the complete blog workflow from the original script with timeout protection
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Blog generation timed out")
            
            # Set a 4-minute timeout (less than gunicorn's 5-minute timeout)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(240)
            
            try:
                complete_workflow = create_complete_blog_workflow(
                    target_keyword=target_keyword,
                    num_competitors=num_competitors
                )
            finally:
                signal.alarm(0)  # Cancel the alarm
            
            # Note: For now, the original script doesn't support custom outlines/length/secondary keywords
            # These could be implemented by modifying the content generation prompts
            # This is a future enhancement opportunity
            
            if "error" in complete_workflow:
                return {
                    'success': False,
                    'error': complete_workflow.get('error'),
                    'content': None
                }
            
            # Extract the results
            blog_post = complete_workflow.get('blog_post', {})
            formatted_post = complete_workflow.get('formatted_post', '')
            content_strategy = complete_workflow.get('content_strategy', {})
            
            # Get word count
            word_count = blog_post.get('blog_post', {}).get('word_count', 'N/A')
            
            return {
                'success': True,
                'error': None,
                'content': {
                    'formatted_post': formatted_post,
                    'blog_post_data': blog_post,
                    'content_strategy': content_strategy,
                    'word_count': word_count,
                    'target_keyword': target_keyword,
                    'competitors_analyzed': complete_workflow.get('workflow_summary', {}).get('competitors_analyzed', 0)
                }
            }
            
        except TimeoutError:
            logger.error("Blog generation timed out")
            return {
                'success': False,
                'error': 'Blog generation timed out. Please try again with a simpler topic or fewer competitors.',
                'content': None
            }
        except Exception as e:
            logger.error(f"Error in blog generation: {str(e)}")
            # Check for specific API errors
            error_message = str(e)
            if "PERMISSION_DENIED" in error_message or "API_KEY_INVALID" in error_message:
                error_message = "Invalid API key. Please check your Gemini API key configuration."
            elif "RESOURCE_EXHAUSTED" in error_message:
                error_message = "API quota exceeded. Please try again later."
            elif "UNAVAILABLE" in error_message:
                error_message = "API service temporarily unavailable. Please try again later."
            elif "SystemExit" in error_message:
                error_message = "Process was terminated due to memory constraints. Try with fewer competitors or simpler topic."
            else:
                error_message = f"Blog generation failed: {error_message}"
                
            return {
                'success': False,
                'error': error_message,
                'content': None
            }
    
    def save_as_document(self, formatted_post, filename):
        """
        Save the formatted blog post as a .docx file
        
        Args:
            formatted_post (str): Formatted blog content
            filename (str): Filename for the document
            
        Returns:
            str: File path where document was saved, or None if failed
        """
        try:
            # Create full file path
            file_path = os.path.join(self.media_root, 'generated_blogs', filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Use the original save_as_docx function if available
            if save_as_docx:
                save_as_docx(formatted_post, file_path)
                logger.info(f"Document saved to: {file_path}")
                return file_path
            else:
                # Fallback: create a simple document
                from docx import Document
                
                doc = Document()
                doc.add_heading("Generated Blog Post", 0)
                
                # Add the content as paragraphs
                for line in formatted_post.split('\n'):
                    if line.strip():
                        if line.startswith('#'):
                            # Handle markdown headings
                            level = len(line) - len(line.lstrip('#'))
                            heading_text = line.strip('#').strip()
                            doc.add_heading(heading_text, level)
                        else:
                            doc.add_paragraph(line)
                
                doc.save(file_path)
                logger.info(f"Document saved to: {file_path}")
                return file_path
                
        except Exception as e:
            logger.error(f"Error saving document: {str(e)}")
            return None
    
    def get_generated_files_dir(self):
        """Get the directory where generated files are stored"""
        return os.path.join(self.media_root, 'generated_blogs')


# Utility function to check if API keys are configured
def check_api_configuration():
    """
    Check if required API keys are configured
    
    Returns:
        dict: Configuration status
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    serp_api_key = os.getenv("SERP_API_KEY")
    google_api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
    
    missing_keys = []
    if not serp_api_key:
        missing_keys.append("SERP_API_KEY")
    if not google_api_key:
        missing_keys.append("GOOGLE_GEMINI_API_KEY")
    
    return {
        'configured': len(missing_keys) == 0,
        'missing_keys': missing_keys,
        'serp_available': bool(serp_api_key),
        'gemini_available': bool(google_api_key)
    }