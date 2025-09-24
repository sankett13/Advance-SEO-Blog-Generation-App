from django import forms
from .models import BlogGeneration


class BlogGenerationForm(forms.ModelForm):
    """Form for blog generation input"""
    
    class Meta:
        model = BlogGeneration
        fields = ['title', 'primary_keywords', 'num_competitors', 'secondary_keywords', 'blog_outline', 'target_length']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., "Best AI Tools for Email Marketing 2025"',
                'required': True
            }),
            'primary_keywords': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'e.g., ai email marketing tools, email automation, marketing ai',
                'required': True
            }),
            'num_competitors': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 1,
                'max': 10,
                'value': 3
            }),
            'secondary_keywords': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 2,
                'placeholder': 'e.g., lead generation, crm integration, sms marketing (Optional)'
            }),
            'blog_outline': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'e.g., 1. Introduction, 2. Top 5 AI Tools, 3. Comparison Table, 4. Conclusion (Optional)'
            }),
            'target_length': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., 1500 words, 2-3 pages, detailed article (Optional)'
            })
        }
        labels = {
            'title': 'Blog Title or Target Topic',
            'primary_keywords': 'Primary Keywords (comma-separated)',
            'num_competitors': 'Number of Competitors to Analyze',
            'secondary_keywords': 'Secondary Keywords (Optional)',
            'blog_outline': 'Custom Blog Outline (Optional)',
            'target_length': 'Target Blog Length (Optional)'
        }
        help_texts = {
            'title': 'Enter the main topic or title for your blog post',
            'primary_keywords': 'Enter relevant keywords separated by commas',
            'num_competitors': 'Choose how many competitor blogs to analyze (1-10)',
            'secondary_keywords': 'Additional keywords to include in the content',
            'blog_outline': 'Specify a custom structure or outline for the blog',
            'target_length': 'Specify desired length or depth of the blog post'
        }