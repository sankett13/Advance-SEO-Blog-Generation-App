import json
from django import template

register = template.Library()

@register.filter
def pprint(value):
    """Pretty print JSON data"""
    if value:
        try:
            return json.dumps(value, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)
    return ""