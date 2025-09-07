"""
YouTube Utilities Module
A clean utility module for YouTube video ID extraction and validation.
"""

import re
from urllib.parse import urlparse, parse_qs

def get_youtube_video_id(url):
    """
    Extract YouTube video ID from various YouTube URL formats.
    
    Supported formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://www.youtube.com/v/VIDEO_ID
    - https://m.youtube.com/watch?v=VIDEO_ID
    
    Args:
        url (str): YouTube video URL
        
    Returns:
        str: Video ID if found, None otherwise
        
    Example:
        >>> get_youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
    """
    
    if not url or not isinstance(url, str):
        return None
        
    # Handle different YouTube URL patterns
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard and other formats
        r'(?:embed\/)([0-9A-Za-z_-]{11})',  # Embed format
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})', # Shortened format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    # Alternative method using urllib.parse for standard URLs
    try:
        parsed_url = urlparse(url)
        if parsed_url.hostname in ['www.youtube.com', 'youtube.com', 'm.youtube.com']:
            if parsed_url.path == '/watch':
                return parse_qs(parsed_url.query).get('v', [None])[0]
            elif parsed_url.path.startswith('/embed/'):
                return parsed_url.path.split('/')[2]
            elif parsed_url.path.startswith('/v/'):
                return parsed_url.path.split('/')[2]
        elif parsed_url.hostname == 'youtu.be':
            return parsed_url.path[1:]  # Remove the leading '/'
    except Exception:
        pass
    
    return None

def validate_youtube_id(video_id):
    """
    Validate if the extracted ID is a valid YouTube video ID format.
    
    Args:
        video_id (str): Video ID to validate
        
    Returns:
        bool: True if valid format, False otherwise
        
    Example:
        >>> validate_youtube_id("dQw4w9WgXcQ")
        True
    """
    if not video_id:
        return False
    
    # YouTube video IDs are 11 characters long and contain letters, numbers, hyphens, and underscores
    pattern = r'^[0-9A-Za-z_-]{11}$'
    return bool(re.match(pattern, video_id))

def extract_and_validate_youtube_id(url):
    """
    Extract and validate YouTube video ID in one step.
    
    Args:
        url (str): YouTube video URL
        
    Returns:
        str: Valid video ID if found and valid, None otherwise
        
    Example:
        >>> extract_and_validate_youtube_id("https://youtu.be/dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
    """
    video_id = get_youtube_video_id(url)
    return video_id if validate_youtube_id(video_id) else None

def is_youtube_url(url):
    """
    Check if a URL is a YouTube URL.
    
    Args:
        url (str): URL to check
        
    Returns:
        bool: True if it's a YouTube URL, False otherwise
        
    Example:
        >>> is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        True
    """
    if not url or not isinstance(url, str):
        return False
        
    try:
        parsed_url = urlparse(url)
        youtube_domains = ['www.youtube.com', 'youtube.com', 'm.youtube.com', 'youtu.be']
        return parsed_url.hostname in youtube_domains
    except Exception:
        return False