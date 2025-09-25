"""
Memory-optimized SEO content automation functions
"""
import os
import gc
from functools import wraps

def memory_efficient_decorator(func):
    """Decorator to clean up memory after function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            # Force garbage collection after heavy operations
            gc.collect()
            return result
        except Exception as e:
            # Clean up memory even if there's an error
            gc.collect()
            raise e
    return wrapper

def truncate_content(text, max_chars=2000):
    """Truncate content to prevent memory overload"""
    if not text or len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"

def optimize_scraped_data(scraped_data, max_content_per_competitor=1500):
    """Reduce the size of scraped data to prevent memory issues"""
    optimized_data = []
    
    for blog in scraped_data:
        if blog:
            # Create a lightweight version of the blog data
            optimized_blog = {
                'url': blog.get('url', ''),
                'title': truncate_content(blog.get('title', ''), 200),
                'metadescription': truncate_content(blog.get('metadescription', ''), 300),
                'h1': [{'heading': truncate_content(h.get('heading', ''), 100)} for h in (blog.get('h1', []) or [])[:3]],
                'h2': [{'heading': truncate_content(h.get('heading', ''), 100)} for h in (blog.get('h2', []) or [])[:5]],
                'h3': [{'heading': truncate_content(h.get('heading', ''), 100)} for h in (blog.get('h3', []) or [])[:3]],
                # Significantly reduce or remove full content
                'content': truncate_content(blog.get('content', ''), max_content_per_competitor)
            }
            optimized_data.append(optimized_blog)
    
    return optimized_data

@memory_efficient_decorator
def memory_efficient_content_gaps_analysis(scraped_blogs_data, paa_questions, target_keyword):
    """
    Memory-efficient version of content gaps analysis
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage
    
    # Optimize the data first
    optimized_data = optimize_scraped_data(scraped_blogs_data, max_content_per_competitor=800)
    
    print("\n🔍 Analyzing Content Gaps with Gemini (Memory Optimized)...")
    
    # Initialize a more memory-efficient model configuration
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # Use Flash instead of Pro for lower memory
        google_api_key=os.getenv("GOOGLE_GEMINI_API_KEY"),
        temperature=0.3,
        max_tokens=2048  # Reduced from 8192
    )
    
    # Create a much more concise summary
    competitors_summary = ""
    for i, blog in enumerate(optimized_data, 1):
        if blog and len(competitors_summary) < 1500:  # Limit total summary size
            competitors_summary += f"Competitor {i}: {blog['title']}\n"
            competitors_summary += f"H2s: {', '.join([h['heading'] for h in blog['h2'][:3]])}\n\n"
    
    # Limit PAA questions
    limited_paa = paa_questions[:5] if paa_questions else []
    paa_summary = "\n".join([f"- {q}" for q in limited_paa])
    
    # Simplified system prompt
    system_prompt = """You are an SEO content strategist. Analyze competitor data and identify 3 key content gaps.
    
    Respond in JSON:
    {
        "content_gaps": [
            {"gap": "brief gap description", "priority": "high/medium/low"}
        ],
        "missing_topics": ["topic1", "topic2", "topic3"],
        "recommended_sections": ["section1", "section2", "section3"]
    }"""
    
    # Much shorter human prompt
    human_prompt = f"""
    Target: "{target_keyword}"
    
    Competitors:
    {competitors_summary[:1000]}
    
    PAA Questions:
    {paa_summary[:500]}
    
    Find 3 content gaps and suggest topics competitors missed."""
    
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = llm.invoke(messages)
        
        # Clean up variables immediately
        del messages, competitors_summary, paa_summary
        gc.collect()
        
        return {
            "success": True,
            "analysis": response.content,
            "competitors_analyzed": len(optimized_data)
        }
        
    except Exception as e:
        print(f"Error in content gap analysis: {str(e)}")
        # Return a fallback response
        return {
            "success": False,
            "error": str(e),
            "fallback_analysis": {
                "content_gaps": [
                    {"gap": "Detailed technical implementation", "priority": "high"},
                    {"gap": "Practical use cases and examples", "priority": "high"},
                    {"gap": "Comparison with alternatives", "priority": "medium"}
                ],
                "missing_topics": ["implementation guide", "best practices", "troubleshooting"],
                "recommended_sections": ["Introduction", "Implementation", "Best Practices", "Conclusion"]
            }
        }