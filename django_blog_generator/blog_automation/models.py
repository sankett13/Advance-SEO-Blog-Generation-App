from django.db import models
from django.utils import timezone


class BlogGeneration(models.Model):
    """Model to store blog generation requests and results"""
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    title = models.CharField(max_length=500, help_text="Blog title or target keyword")
    primary_keywords = models.TextField(help_text="Primary keywords (comma-separated)")
    num_competitors = models.IntegerField(default=3, help_text="Number of competitors to analyze")
    
    # Optional fields for enhanced control
    secondary_keywords = models.TextField(blank=True, null=True, help_text="Secondary keywords (comma-separated) - Optional")
    blog_outline = models.TextField(blank=True, null=True, help_text="Custom blog outline/structure - Optional")
    target_length = models.CharField(max_length=20, blank=True, null=True, help_text="Target blog length - Optional")
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    # Results
    generated_content = models.TextField(blank=True, null=True, help_text="Generated blog content")
    formatted_post = models.TextField(blank=True, null=True, help_text="Formatted blog post")
    word_count = models.CharField(max_length=50, blank=True, null=True)
    
    # Analysis results (stored as JSON)
    competitor_analysis = models.JSONField(blank=True, null=True)
    blog_analysis_data = models.JSONField(blank=True, null=True, help_text="Generated blog analysis and outline data")
    
    # File path for generated document
    doc_file_path = models.CharField(max_length=500, blank=True, null=True)
    
    # Metadata
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Error handling
    error_message = models.TextField(blank=True, null=True)
    
    class Meta:
        ordering = ['-created_at']
        
    def __str__(self):
        return f"Blog: {self.title[:50]}... ({self.status})"
    
    @property
    def filename(self):
        """Generate filename for the document"""
        import re
        clean_title = re.sub(r'[^\w\s-]', '', self.title)
        clean_title = re.sub(r'[-\s]+', '_', clean_title)
        timestamp = self.created_at.strftime('%Y%m%d_%H%M%S')
        return f"blog_post_{timestamp}_{clean_title[:30]}.docx"