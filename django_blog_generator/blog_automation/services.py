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
                    'error': 'GOOGLE_GEMINI_API_KEY environment variable is not set. Please set it in your environment or .env file.',
                    'content': None
                }
                
            if not os.getenv('SERP_API_KEY'):
                return {
                    'success': False,
                    'error': 'SERP_API_KEY environment variable is not set. Please set it in your environment or .env file.',
                    'content': None
                }
            
            # Prepare the target keyword (use title as primary target)
            target_keyword = title
            
            logger.info(f"Starting blog generation for: {target_keyword}")
            logger.info(f"Number of competitors to analyze: {num_competitors}")
            logger.info(f"Optional parameters - Secondary keywords: {secondary_keywords}, Outline: {blog_outline}, Length: {target_length}")
            
            # Note: The original create_complete_blog_workflow doesn't support the optional parameters yet
            # This is a future enhancement opportunity to modify the prompts with custom outlines/length/secondary keywords
            
            # Check if the original functions are available
            if not create_complete_blog_workflow:
                return {
                    'success': False,
                    'error': 'SEO automation functions not available. Please check imports.',
                    'content': None
                }
            
            # Use ultra-minimal approach for Render's memory constraints
            import gc
            
            # Force garbage collection before starting
            gc.collect()
            
            try:
                # Use full competitor analysis workflow for local development
                logger.info("Using full competitor analysis workflow")
                complete_workflow = create_complete_blog_workflow(
                    target_keyword=target_keyword,
                    num_competitors=num_competitors
                )
            finally:
                # Clean up memory
                gc.collect()
            
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
            elif "ImportError" in error_message:
                error_message = "Missing dependencies. Please ensure all required packages are installed."
            elif "ConnectTimeout" in error_message or "ReadTimeout" in error_message:
                error_message = "Network timeout. Please check your internet connection and try again."
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
    
    def _memory_efficient_blog_workflow(self, target_keyword, num_competitors=2):
        """
        Memory-efficient version of the blog workflow for Render deployment
        """
        import gc
        from memory_optimized_seo import memory_efficient_content_gaps_analysis
        
        try:
            # Step 1: Get competitors (limited to reduce memory usage)
            print(f"\n1. 🔍 Getting top {num_competitors} competitors...")
            
            # Use the original functions but with reduced scope
            from seo_content_automation import (
                get_top_5_links_and_paa,
                scrape_blog,
                format_blog_post_for_publication
            )
            
            # Get SERP results using the correct function
            competitor_links, paa_questions = get_top_5_links_and_paa(target_keyword, os.getenv("SERP_API_KEY"), num_results=num_competitors)
            competitors = [{'url': link} for link in competitor_links[:num_competitors]]
            
            print(f"✅ Found {len(competitors)} competitors")
            
            # Step 2: Scrape competitor content (with memory optimization)
            print(f"\n2. 📊 Scraping competitor blog content...")
            scraped_data = []
            
            for i, competitor in enumerate(competitors, 1):
                competitor_url = competitor.get('url', competitor.get('link', ''))
                print(f"   Scraping competitor {i}: {competitor_url}")
                try:
                    blog_data = scrape_blog(competitor_url)
                    if blog_data:
                        scraped_data.append(blog_data)
                        print(f"   ✅ Successfully scraped competitor {i}")
                    else:
                        print(f"   ⚠️ Failed to scrape competitor {i}")
                except Exception as e:
                    print(f"   ⚠️ Error scraping competitor {i}: {str(e)}")
                
                # Force garbage collection after each scrape
                gc.collect()
            
            # Step 3: Memory-efficient content analysis
            print(f"\n3. 🤖 Running memory-efficient AI analysis...")
            content_gaps = memory_efficient_content_gaps_analysis(
                scraped_data, paa_questions, target_keyword
            )
            
            # Clean up scraped data to free memory
            del scraped_data
            gc.collect()
            
            # Step 4: Generate blog post (with reduced complexity)
            print(f"\n4. ✍️ Generating blog post...")
            
            # Simplified blog generation prompt
            blog_result = self._generate_simple_blog_post(target_keyword, content_gaps)
            
            # Step 5: Format the post
            print(f"\n5. 📝 Formatting blog post...")
            formatted_post = format_blog_post_for_publication(
                blog_result.get('blog_post', {}), 
                blog_result.get('seo_meta', {})
            )
            
            return {
                'blog_post': blog_result,
                'formatted_post': formatted_post,
                'content_strategy': content_gaps,
                'workflow_summary': {
                    'competitors_analyzed': len(competitors),
                    'memory_optimized': True
                }
            }
            
        except Exception as e:
            print(f"Error in memory-efficient workflow: {str(e)}")
            raise e
    
    def _generate_simple_blog_post(self, target_keyword, content_gaps):
        """Generate a simple blog post with reduced memory usage"""
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SystemMessage
        import json
        import gc
        
        # Use Flash model for lower memory usage
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_GEMINI_API_KEY"),
            temperature=0.3,
            max_tokens=3000  # Reduced token limit
        )
        
        system_prompt = """You are an expert blog writer. Create a comprehensive, SEO-optimized blog post.
        
        Return JSON format:
        {
            "blog_post": {
                "title": "SEO optimized title",
                "content": "Full blog post content in markdown",
                "word_count": number
            },
            "seo_meta": {
                "meta_title": "title for meta tag",
                "meta_description": "description for meta tag"
            }
        }"""
        
        # Get content gaps info
        gaps_info = ""
        if content_gaps.get('success') and content_gaps.get('analysis'):
            gaps_info = content_gaps['analysis'][:500]  # Limit size
        elif content_gaps.get('fallback_analysis'):
            fallback = content_gaps['fallback_analysis']
            gaps_info = f"Key topics to cover: {', '.join(fallback['missing_topics'][:3])}"
        
        human_prompt = f"""
        Write a comprehensive blog post about: "{target_keyword}"
        
        Content guidance: {gaps_info}
        
        Requirements:
        - 1500-2000 words
        - Include H2 and H3 headings
        - SEO optimized
        - Practical and informative
        - Include introduction and conclusion
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = llm.invoke(messages)
            
            # Clean up
            del messages
            gc.collect()
            
            # Parse the response
            try:
                result = json.loads(response.content)
                return result
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "blog_post": {
                        "title": f"Complete Guide to {target_keyword}",
                        "content": response.content,
                        "word_count": len(response.content.split())
                    },
                    "seo_meta": {
                        "meta_title": f"{target_keyword} - Complete Guide",
                        "meta_description": f"Comprehensive guide to {target_keyword}. Learn everything you need to know."
                    }
                }
                
        except Exception as e:
            print(f"Error generating blog post: {str(e)}")
            raise e

    def _ultra_minimal_blog_generation(self, target_keyword, secondary_keywords=None, blog_outline=None, target_length=None):
        """
        Ultra-minimal blog generation that avoids all memory-intensive operations
        Perfect for Render's free tier constraints
        """
        import gc
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SystemMessage
        import json
        
        print(f"\n🚀 Ultra-Minimal Blog Generation for: {target_keyword}")
        print("⚡ Optimized for low memory usage - no competitor analysis")
        
        try:
            # Use the most memory-efficient model
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=os.getenv("GOOGLE_GEMINI_API_KEY"),
                temperature=0.3,
                max_tokens=2048,  # Further reduced
                max_retries=1  # Reduce retries
            )
            
            # Build context from user inputs only (no external API calls)
            user_context = ""
            if secondary_keywords:
                user_context += f"\nSecondary Keywords: {secondary_keywords}"
            if blog_outline:
                user_context += f"\nSuggested Outline: {blog_outline}"
            if target_length:
                user_context += f"\nTarget Length: {target_length}"
            
            # Ultra-simple system prompt - avoid JSON to reduce parsing issues
            system_prompt = """You are an expert blog writer. Create a comprehensive, SEO-optimized blog post in markdown format. Start with a clear title and write informative content."""
            
            # Simple human prompt - no complex analysis
            human_prompt = f"""Write a comprehensive blog post about: "{target_keyword}"
            
            {user_context}
            
            Requirements:
            - Start with a clear title (use # for main title)
            - 1200-1800 words
            - Use ## for H2 headings and ### for H3 headings  
            - SEO optimized for the keyword
            - Include introduction and conclusion
            - Practical and informative
            - Write everything in markdown format
            
            Please write the complete blog post now."""
            
            # Generate the blog post
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            try:
                response = llm.invoke(messages)
                
                # Immediate cleanup
                del messages, system_prompt, human_prompt
                gc.collect()
                
            except Exception as api_error:
                print(f"⚠️ Gemini API error: {str(api_error)}")
                # Cleanup and fall back to emergency generation
                del messages, system_prompt, human_prompt
                gc.collect()
                raise api_error
            
            # Parse response - expect markdown content directly
            try:
                # Safe extraction of content
                if hasattr(response, 'content'):
                    response_content = response.content
                elif hasattr(response, 'text'):
                    response_content = response.text
                else:
                    response_content = str(response)
                
                # Ensure we have a string
                if not isinstance(response_content, str):
                    response_content = str(response_content)
                
            except Exception as parse_error:
                print(f"⚠️ Error parsing response: {str(parse_error)}")
                response_content = f"Error parsing response for {target_keyword}"
            
            # Extract title from markdown if it exists
            title = f"Complete Guide to {target_keyword}"
            content = response_content
            
            # Try to extract title from first line if it starts with #
            try:
                lines = response_content.split('\n')
                if lines and len(lines) > 0 and lines[0].startswith('# '):
                    title = lines[0][2:].strip()  # Remove '# ' prefix
                    content = '\n'.join(lines[1:]).strip()  # Rest is content
            except Exception:
                # If title extraction fails, use the default
                pass
            
            # Create the structure expected by format_blog_post_for_publication
            # Don't use .get() since we're creating the dict ourselves
            blog_post_data = {
                "blog_post": {
                    "title": title,
                    "content": content,
                    "word_count": len(response_content.split()) if response_content else 0
                },
                "seo_meta": {
                    "meta_title": f"{title} - Complete Guide",
                    "meta_description": f"Learn everything about {target_keyword} in this comprehensive guide."
                }
            }
            
            # Format the post using the original function
            from seo_content_automation import format_blog_post_for_publication
            formatted_post = format_blog_post_for_publication(blog_post_data)
            
            return {
                'blog_post': blog_post_data,
                'formatted_post': formatted_post,
                'content_strategy': {
                    'success': True,
                    'analysis': 'Generated using ultra-minimal approach for memory efficiency'
                },
                'workflow_summary': {
                    'competitors_analyzed': 0,
                    'memory_optimized': True,
                    'ultra_minimal': True
                }
            }
            
        except Exception as e:
            print(f"Error in ultra-minimal generation: {str(e)}")
            # Ultimate fallback - create basic content
            return self._create_emergency_fallback_blog(target_keyword, secondary_keywords)
    
    def _create_emergency_fallback_blog(self, target_keyword, secondary_keywords=None):
        """
        Emergency fallback that creates basic blog content without any API calls
        """
        print("🆘 Using emergency fallback generation")
        
        # Create basic blog structure
        title = f"Complete Guide to {target_keyword}"
        
        # Basic content template
        content = f"""# {title}

## Introduction

{target_keyword} is an important topic that deserves comprehensive coverage. In this guide, we'll explore everything you need to know about {target_keyword}.

## What is {target_keyword}?

{target_keyword} represents a significant area of interest for many people. Understanding the fundamentals is crucial for success.

## Key Benefits of {target_keyword}

1. **Improved Understanding**: Learn the core concepts
2. **Practical Application**: Apply knowledge effectively  
3. **Best Practices**: Follow proven methods
4. **Avoid Common Mistakes**: Learn from others' experiences

## Getting Started with {target_keyword}

### Step 1: Foundation
Start by understanding the basics of {target_keyword}.

### Step 2: Planning
Create a clear plan for implementing {target_keyword}.

### Step 3: Implementation
Put your knowledge into practice.

## Advanced Concepts

For those ready to go deeper, consider these advanced aspects of {target_keyword}.

## Best Practices

1. Always start with solid fundamentals
2. Practice regularly
3. Stay updated with latest developments
4. Learn from experts in the field

## Common Challenges and Solutions

### Challenge 1: Getting Started
**Solution**: Begin with small, manageable steps.

### Challenge 2: Staying Consistent
**Solution**: Create a regular routine.

## Conclusion

{target_keyword} is a valuable skill worth developing. By following the guidelines in this post, you'll be well on your way to mastery.

Remember to practice regularly and stay committed to continuous learning."""

        # Add secondary keywords if provided
        if secondary_keywords:
            content += f"\n\n## Related Topics\n\nExplore these related areas: {secondary_keywords}"

        blog_post = {
            "blog_post": {
                "title": title,
                "content": content,
                "word_count": len(content.split())
            }
        }
        
        seo_meta = {
            "meta_title": title,
            "meta_description": f"Complete guide to {target_keyword}. Learn everything you need to know."
        }
        
        return {
            'blog_post': blog_post,
            'formatted_post': content,
            'content_strategy': {
                'success': True,
                'analysis': 'Emergency fallback content generated'
            },
            'workflow_summary': {
                'competitors_analyzed': 0,
                'memory_optimized': True,
                'emergency_fallback': True
            }
        }

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