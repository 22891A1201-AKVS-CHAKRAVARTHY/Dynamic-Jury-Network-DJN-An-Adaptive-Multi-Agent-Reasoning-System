from django import template

register = template.Library()


@register.filter
def get_item(d: dict, key: str):
    """Safely get d[key] in Django templates."""
    try:
        if not isinstance(d, dict):
            return ""
        return d.get(key, "")
    except Exception:
        return ""
