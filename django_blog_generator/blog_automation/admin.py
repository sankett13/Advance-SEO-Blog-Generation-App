from django.contrib import admin
from .models import BlogGeneration


@admin.register(BlogGeneration)
class BlogGenerationAdmin(admin.ModelAdmin):
    list_display = ['title', 'status', 'word_count', 'created_at', 'updated_at']
    list_filter = ['status', 'created_at', 'num_competitors']
    search_fields = ['title', 'primary_keywords']
    readonly_fields = ['created_at', 'updated_at', 'filename']
    
    fieldsets = (
        ('Blog Information', {
            'fields': ('title', 'primary_keywords', 'num_competitors')
        }),
        ('Generation Status', {
            'fields': ('status', 'error_message')
        }),
        ('Results', {
            'fields': ('word_count', 'doc_file_path', 'formatted_post'),
            'classes': ['collapse']
        }),
        ('Analysis Data', {
            'fields': ('competitor_analysis', 'blog_outline'),
            'classes': ['collapse']
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at', 'filename'),
            'classes': ['collapse']
        })
    )
    
    def get_readonly_fields(self, request, obj=None):
        readonly = list(self.readonly_fields)
        if obj and obj.status in ['completed', 'processing']:
            readonly.extend(['title', 'primary_keywords', 'num_competitors'])
        return readonly