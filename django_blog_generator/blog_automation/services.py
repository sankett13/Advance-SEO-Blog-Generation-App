"""
SEO Content Automation Service
Integrates the existing seo_content_automation.py functionality into Django
"""
import os
import sys
import json
import logging
import re
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
                
                # Create a modified version that supports target length and other parameters
                complete_workflow = self._enhanced_blog_workflow(
                    target_keyword=target_keyword,
                    num_competitors=num_competitors,
                    secondary_keywords=secondary_keywords,
                    blog_outline=blog_outline,
                    target_length=target_length
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
        Save the formatted blog post as a .docx file with proper formatting (no markdown)
        
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
            
            # Use enhanced document creation with proper formatting
            self._create_enhanced_docx(formatted_post, file_path)
            logger.info(f"Document saved to: {file_path}")
            return file_path
                
        except Exception as e:
            logger.error(f"Error saving document: {str(e)}")
            return None
    
    def _create_enhanced_docx(self, content, file_path):
        """
        Create a DOCX file with proper formatting, converting markdown to Word format
        """
        from docx import Document
        from docx.shared import Inches, Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml.shared import OxmlElement, qn
        from docx.oxml.ns import nsdecls
        from docx.oxml import parse_xml
        import re
        
        doc = Document()
        
        # Set document margins
        sections = doc.sections
        for section in sections:
            section.top_margin = Inches(1)
            section.bottom_margin = Inches(1)
            section.left_margin = Inches(1)
            section.right_margin = Inches(1)
        
        lines = content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
                
            # Handle headings
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                heading_text = line.strip('#').strip()
                
                # Add heading with appropriate level
                if level == 1:
                    heading = doc.add_heading(heading_text, 0)
                elif level == 2:
                    heading = doc.add_heading(heading_text, 1)
                elif level == 3:
                    heading = doc.add_heading(heading_text, 2)
                else:
                    heading = doc.add_heading(heading_text, 3)
                    
            # Handle tables
            elif line.startswith('|') and '|' in line:
                table_lines = []
                # Collect all table lines
                while i < len(lines) and lines[i].strip().startswith('|'):
                    table_line = lines[i].strip()
                    if not table_line.replace('|', '').replace('-', '').replace(' ', ''):
                        # Skip separator line
                        i += 1
                        continue
                    table_lines.append(table_line)
                    i += 1
                
                if table_lines:
                    self._add_table_to_doc(doc, table_lines)
                continue
                
            # Handle lists
            elif line.startswith(('-', '*', '+')):
                # Handle unordered list
                list_text = line[1:].strip()
                list_text = self._clean_markdown_formatting(list_text)
                p = doc.add_paragraph()
                p.style = 'List Bullet'
                p.add_run(list_text)
                
            elif re.match(r'^\d+\.', line):
                # Handle ordered list
                list_text = re.sub(r'^\d+\.\s*', '', line)
                list_text = self._clean_markdown_formatting(list_text)
                p = doc.add_paragraph()
                p.style = 'List Number'
                p.add_run(list_text)
                
            # Handle regular paragraphs
            else:
                if line:
                    # Clean markdown and add paragraph
                    clean_text = self._clean_markdown_formatting(line)
                    p = doc.add_paragraph()
                    self._add_formatted_text(p, clean_text)
            
            i += 1
        
        doc.save(file_path)
    
    def _add_table_to_doc(self, doc, table_lines):
        """Add a professionally formatted table to the document"""
        from docx.shared import Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        
        if not table_lines:
            return
            
        # Parse table data
        rows_data = []
        for line in table_lines:
            cells = [cell.strip() for cell in line.split('|')[1:-1]]  # Remove empty first/last
            if cells:
                rows_data.append(cells)
        
        if not rows_data:
            return
            
        # Create table with proper styling
        table = doc.add_table(rows=len(rows_data), cols=len(rows_data[0]))
        
        # Apply built-in professional table style
        table.style = 'Medium Grid 1 Accent 1'
        
        # Fill table data with enhanced formatting
        for row_idx, row_data in enumerate(rows_data):
            row = table.rows[row_idx]
            
            for col_idx, cell_data in enumerate(row_data):
                if col_idx < len(row.cells):
                    cell = row.cells[col_idx]
                    
                    # Clean and add formatted text
                    clean_text = self._clean_markdown_formatting(cell_data)
                    
                    # Clear existing content and add new
                    cell.text = clean_text
                    
                    # Enhanced formatting for cells
                    paragraph = cell.paragraphs[0]
                    
                    # Header row styling
                    if row_idx == 0:
                        for run in paragraph.runs:
                            run.bold = True
                            if hasattr(run.font, 'size'):
                                run.font.size = Pt(11)
                        # Center align headers
                        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    else:
                        for run in paragraph.runs:
                            if hasattr(run.font, 'size'):
                                run.font.size = Pt(10)
                        # Left align content
                        paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
        
        # Set table properties for better appearance
        try:
            table.autofit = True
        except:
            pass  # Some versions don't support autofit
        
        # Add space after table
        doc.add_paragraph()
    
    def _clean_markdown_formatting(self, text):
        """Remove markdown formatting and return clean text"""
        # Remove markdown bold/italic
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold**
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *italic*
        text = re.sub(r'__([^_]+)__', r'\1', text)      # __bold__
        text = re.sub(r'_([^_]+)_', r'\1', text)        # _italic_
        
        # Handle code blocks
        text = re.sub(r'`([^`]+)`', r'\1', text)        # `code`
        
        # Remove markdown links but keep the text and URL
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', text)
        
        return text
    
    def _add_formatted_text(self, paragraph, text):
        """Add formatted text to paragraph with hyperlinks"""
        import re
        
        # Find hyperlinks in format [text](url) 
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        
        last_end = 0
        for match in re.finditer(link_pattern, text):
            # Add text before link
            if match.start() > last_end:
                paragraph.add_run(text[last_end:match.start()])
            
            # Add hyperlink
            link_text = match.group(1)
            link_url = match.group(2)
            self._add_hyperlink(paragraph, link_url, link_text)
            
            last_end = match.end()
        
        # Add remaining text
        if last_end < len(text):
            paragraph.add_run(text[last_end:])
    
    def _add_hyperlink(self, paragraph, url, text):
        """Add a hyperlink to a paragraph"""
        from docx.oxml.shared import OxmlElement, qn
        
        # Create hyperlink element
        hyperlink = OxmlElement('w:hyperlink')
        hyperlink.set(qn('w:anchor'), url)
        
        # Create run for the link
        run = OxmlElement('w:r')
        rPr = OxmlElement('w:rPr')
        
        # Add blue color and underline for hyperlink appearance
        color = OxmlElement('w:color')
        color.set(qn('w:val'), '0000FF')
        rPr.append(color)
        
        u = OxmlElement('w:u')
        u.set(qn('w:val'), 'single')
        rPr.append(u)
        
        run.append(rPr)
        
        # Add text
        t = OxmlElement('w:t')
        t.text = text
        run.append(t)
        
        hyperlink.append(run)
        paragraph._p.append(hyperlink)
    
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

    def _enhanced_blog_workflow(self, target_keyword, num_competitors=3, secondary_keywords=None, blog_outline=None, target_length=None):
        """
        Enhanced blog workflow that supports target length and other custom parameters
        """
        from seo_content_automation import (
            create_complete_content_strategy,
            generate_blog_outline,
            generate_blog_post,
            format_blog_post_for_publication
        )
        
        logger.info(f"Enhanced workflow with target_length: {target_length}")
        
        # Step 1: Create content strategy (competitor analysis)
        print(f"\n📊 STEP 1: CREATING CONTENT STRATEGY")
        complete_strategy = create_complete_content_strategy(target_keyword, num_competitors)
        
        if "error" in complete_strategy:
            return {"error": f"Content strategy creation failed: {complete_strategy['error']}"}
        
        # Step 2: Generate blog outline with custom parameters
        print(f"\n📝 STEP 2: BLOG OUTLINE GENERATION")
        
        # Modify the outline generation to include custom parameters
        outline_data = self._generate_custom_outline(complete_strategy, blog_outline, target_length, secondary_keywords)
        
        if "error" in outline_data:
            return {"error": f"Outline generation failed: {outline_data['error']}"}
        
        # Step 3: Generate blog post with custom parameters
        print(f"\n📝 STEP 3: GENERATING BLOG POST")
        blog_post = self._generate_enhanced_blog_post(outline_data, complete_strategy, target_length)
        
        if "error" in blog_post:
            return {"error": f"Blog post generation failed: {blog_post['error']}"}
        
        # Step 4: Format for publication
        print(f"\n📋 STEP 4: FORMATTING FOR PUBLICATION")
        formatted_post = format_blog_post_for_publication(blog_post)
        
        # Return complete workflow result
        return {
            "target_keyword": target_keyword,
            "creation_date": datetime.now().isoformat(),
            "content_strategy": complete_strategy,
            "blog_post": blog_post,
            "formatted_post": formatted_post,
            "workflow_summary": {
                "competitors_analyzed": complete_strategy.get("strategy_summary", {}).get("competitors_analyzed", 0),
                "outline_generated": True,
                "blog_post_created": True,
                "estimated_word_count": blog_post.get("blog_post", {}).get("word_count", "N/A"),
                "custom_parameters_used": {
                    "target_length": target_length,
                    "secondary_keywords": secondary_keywords,
                    "custom_outline": blog_outline is not None
                }
            }
        }
    
    def _generate_custom_outline(self, strategy_data, custom_outline, target_length, secondary_keywords):
        """Generate outline with custom parameters"""
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SystemMessage
        import json
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=os.getenv("GOOGLE_GEMINI_API_KEY"),
            temperature=0.3,
            max_tokens=8192
        )
        
        # Build custom context
        target_word_count = "2500-3500 words"  # default
        if target_length:
            if any(word in target_length.lower() for word in ['short', '1000', '1500']):
                target_word_count = "1000-1500 words"
            elif any(word in target_length.lower() for word in ['medium', '2000', '2500']):
                target_word_count = "2000-2500 words"
            elif any(word in target_length.lower() for word in ['long', '3000', '4000']):
                target_word_count = "3000-4000 words"
            elif any(char.isdigit() for char in target_length):
                target_word_count = target_length + " words"
        
        custom_requirements = ""
        if secondary_keywords:
            custom_requirements += f"\nSecondary Keywords to Include: {secondary_keywords}"
        if custom_outline:
            custom_requirements += f"\nCustom Outline Requirements: {custom_outline}"
        
        system_prompt = f"""You are an expert content strategist. Create a detailed blog outline based on competitor analysis.
        
        Target Word Count: {target_word_count}
        {custom_requirements}
        
        Return JSON format:
        {{
            "content_structure": {{
                "estimated_word_count": "{target_word_count}",
                "sections": [
                    {{
                        "section_title": "Introduction",
                        "word_count": "200-300",
                        "key_points": ["point1", "point2"]
                    }}
                ]
            }}
        }}"""
        
        # Get content gaps from strategy
        content_gaps = strategy_data.get('content_gaps_analysis', {})
        gaps_summary = str(content_gaps)[:1000]  # Limit size
        
        human_prompt = f"""Based on this competitor analysis, create an optimized blog outline:
        
        Content Gaps Found:
        {gaps_summary}
        
        Requirements:
        - Target length: {target_word_count}
        - Include sections that address content gaps
        - SEO optimized structure
        {custom_requirements}
        """
        
        try:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
            response = llm.invoke(messages)
            
            # Parse response
            outline_data = json.loads(response.content)
            return outline_data
            
        except Exception as e:
            logger.error(f"Error generating custom outline: {str(e)}")
            # Fallback to original outline generation
            from seo_content_automation import generate_blog_outline
            return generate_blog_outline(strategy_data)
    
    def _generate_enhanced_blog_post(self, outline_data, analysis_results, target_length):
        """Generate blog post with enhanced formatting and hyperlinks"""
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SystemMessage
        import json
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=os.getenv("GOOGLE_GEMINI_API_KEY"),
            temperature=0.3,
            max_tokens=8192
        )
        
        # Get comprehensive tool links for context
        tool_links_context = self._get_tool_links_context()
        
        # Extract outline structure
        content_structure = outline_data.get("content_structure", {})
        sections = content_structure.get("sections", [])
        target_word_count = content_structure.get("estimated_word_count", "2500-3500 words")
        
        # Override target word count if specified
        if target_length:
            target_word_count = target_length
        
        system_prompt = f"""You are an expert blog writer. Create comprehensive, SEO-optimized content with proper formatting.

        IMPORTANT FORMATTING REQUIREMENTS:
        1. ALWAYS add clickable links when mentioning tools, platforms, or services
        2. Use this format: [Tool Name](URL) - for example: [ChatGPT](https://chat.openai.com)
        3. Create well-formatted tables using markdown table syntax with proper headers
        4. Include relevant external links to authoritative sources
        5. Target word count: {target_word_count}
        6. Use engaging, informative tone

        AVAILABLE TOOL LINKS (use these when mentioning tools):
        {tool_links_context}
        
        Return JSON:
        {{
            "blog_post": {{
                "title": "SEO optimized title",
                "content": "Full blog content with proper formatting and clickable links",
                "word_count": "actual word count"
            }},
            "seo_meta": {{
                "meta_title": "meta title",
                "meta_description": "meta description"
            }}
        }}"""
        
        # Build detailed prompt with sections
        sections_text = ""
        for section in sections:
            sections_text += f"- {section.get('section_title', 'Section')}: {section.get('word_count', '300-400')} words\n"
            key_points = section.get('key_points', [])
            if key_points:
                sections_text += f"  Key points: {', '.join(key_points[:3])}\n"
        
        human_prompt = f"""Create a comprehensive blog post following this structure:
        
        SECTIONS TO INCLUDE:
        {sections_text}
        
        REQUIREMENTS:
        - Target length: {target_word_count}
        - When mentioning any tools, platforms, or services, format them as clickable links using the URLs provided above
        - Example: "You can use [ChatGPT](https://chat.openai.com) for content generation"
        - Create comparison tables with proper markdown formatting when comparing tools or features
        - Include relevant external links to authoritative sources and official websites
        - SEO optimized for the main keyword
        - Professional, engaging tone with actionable insights
        - Use tables for data comparisons, tool features, or step-by-step processes
        """
        
        try:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
            response = llm.invoke(messages)
            
            # Parse response
            blog_data = json.loads(response.content)
            
            # Post-process content to add any missed links
            if "blog_post" in blog_data and "content" in blog_data["blog_post"]:
                blog_data["blog_post"]["content"] = self._add_automatic_links(blog_data["blog_post"]["content"])
            
            return blog_data
            
        except Exception as e:
            logger.error(f"Error generating enhanced blog post: {str(e)}")
            # Fallback to original blog generation
            from seo_content_automation import generate_blog_post
            return generate_blog_post(outline_data, analysis_results)
    
    def _get_tool_links_context(self):
        """Get formatted context of available tool links for AI generation"""
        tool_map = self._get_comprehensive_tool_mapping()
        
        context_lines = []
        for category, tools in tool_map.items():
            context_lines.append(f"\n{category.upper()}:")
            for tool_name, url in tools.items():
                context_lines.append(f"- {tool_name}: {url}")
        
        return "\n".join(context_lines)
    
    def _get_comprehensive_tool_mapping(self):
        """Comprehensive mapping of tool names to their URLs"""
        return {
            "AI_TOOLS": {
                "ChatGPT": "https://chat.openai.com",
                "Claude": "https://claude.ai",
                "Gemini": "https://gemini.google.com",
                "Copilot": "https://copilot.microsoft.com",
                "GitHub Copilot": "https://github.com/features/copilot",
                "Midjourney": "https://midjourney.com",
                "DALL-E": "https://openai.com/dall-e",
                "Stable Diffusion": "https://stability.ai",
                "Jasper": "https://jasper.ai",
                "Copy.ai": "https://copy.ai",
                "Writesonic": "https://writesonic.com",
                "Grammarly": "https://grammarly.com",
                "QuillBot": "https://quillbot.com"
            },
            "CODING_TOOLS": {
                "Visual Studio Code": "https://code.visualstudio.com",
                "VS Code": "https://code.visualstudio.com",
                "PyCharm": "https://jetbrains.com/pycharm",
                "IntelliJ IDEA": "https://jetbrains.com/idea",
                "Sublime Text": "https://sublimetext.com",
                "Atom": "https://atom.io",
                "Vim": "https://vim.org",
                "Emacs": "https://gnu.org/software/emacs",
                "WebStorm": "https://jetbrains.com/webstorm",
                "Android Studio": "https://developer.android.com/studio",
                "Xcode": "https://developer.apple.com/xcode"
            },
            "DEV_PLATFORMS": {
                "GitHub": "https://github.com",
                "GitLab": "https://gitlab.com",
                "Bitbucket": "https://bitbucket.org",
                "Stack Overflow": "https://stackoverflow.com",
                "CodePen": "https://codepen.io",
                "JSFiddle": "https://jsfiddle.net",
                "Repl.it": "https://replit.com",
                "Glitch": "https://glitch.com",
                "Heroku": "https://heroku.com",
                "Netlify": "https://netlify.com",
                "Vercel": "https://vercel.com",
                "AWS": "https://aws.amazon.com",
                "Google Cloud": "https://cloud.google.com",
                "Azure": "https://azure.microsoft.com"
            },
            "DESIGN_TOOLS": {
                "Figma": "https://figma.com",
                "Adobe XD": "https://adobe.com/products/xd.html",
                "Sketch": "https://sketch.com",
                "Canva": "https://canva.com",
                "Adobe Photoshop": "https://adobe.com/products/photoshop.html",
                "Adobe Illustrator": "https://adobe.com/products/illustrator.html",
                "InVision": "https://invisionapp.com",
                "Framer": "https://framer.com",
                "Principle": "https://principleformac.com"
            },
            "FRAMEWORKS": {
                "React": "https://reactjs.org",
                "Vue.js": "https://vuejs.org",
                "Angular": "https://angular.io",
                "Django": "https://djangoproject.com",
                "Flask": "https://flask.palletsprojects.com",
                "Express.js": "https://expressjs.com",
                "Next.js": "https://nextjs.org",
                "Nuxt.js": "https://nuxtjs.org",
                "Laravel": "https://laravel.com",
                "Ruby on Rails": "https://rubyonrails.org",
                "Spring Boot": "https://spring.io/projects/spring-boot"
            },
            "DATABASES": {
                "MongoDB": "https://mongodb.com",
                "PostgreSQL": "https://postgresql.org",
                "MySQL": "https://mysql.com",
                "Redis": "https://redis.io",
                "Firebase": "https://firebase.google.com",
                "Supabase": "https://supabase.com",
                "PlanetScale": "https://planetscale.com"
            },
            "PRODUCTIVITY": {
                "Notion": "https://notion.so",
                "Obsidian": "https://obsidian.md",
                "Roam Research": "https://roamresearch.com",
                "Trello": "https://trello.com",
                "Asana": "https://asana.com",
                "Jira": "https://atlassian.com/software/jira",
                "Slack": "https://slack.com",
                "Discord": "https://discord.com",
                "Microsoft Teams": "https://microsoft.com/en-us/microsoft-teams",
                "Zoom": "https://zoom.us"
            },
            "ANALYTICS": {
                "Google Analytics": "https://analytics.google.com",
                "Mixpanel": "https://mixpanel.com",
                "Amplitude": "https://amplitude.com",
                "Hotjar": "https://hotjar.com",
                "Google Tag Manager": "https://tagmanager.google.com"
            }
        }
    
    def _add_automatic_links(self, content):
        """Automatically add links to mentioned tools in content"""
        tool_map = self._get_comprehensive_tool_mapping()
        
        # Flatten the tool mapping for easier search
        all_tools = {}
        for category, tools in tool_map.items():
            all_tools.update(tools)
        
        # Sort by length (longest first) to avoid partial matches
        sorted_tools = sorted(all_tools.items(), key=lambda x: len(x[0]), reverse=True)
        
        modified_content = content
        
        for tool_name, url in sorted_tools:
            # Create case-insensitive pattern that doesn't match already linked text
            pattern = rf'(?<!\[)(?<!\]\()({re.escape(tool_name)})(?!\]\()'
            
            # Replace only if the tool is mentioned but not already linked
            def replace_match(match):
                return f"[{match.group(1)}]({url})"
            
            modified_content = re.sub(pattern, replace_match, modified_content, flags=re.IGNORECASE)
        
        return modified_content

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