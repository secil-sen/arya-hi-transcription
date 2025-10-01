"""
Cross-platform path utilities for clean file handling.
"""

import os
from pathlib import Path
from typing import Union, List


def safe_join(*paths: str) -> str:
    """
    Safely join paths using os.path.join for cross-platform compatibility.

    Args:
        *paths: Path components to join

    Returns:
        str: Properly joined path for current platform
    """
    return os.path.join(*paths)


def normalize_path(path: Union[str, Path]) -> str:
    """
    Normalize path to use correct separators for current platform.

    Args:
        path: Input path

    Returns:
        str: Normalized path
    """
    return os.path.normpath(str(path))


def create_directory_safely(directory: str) -> str:
    """
    Create directory if it doesn't exist, handling cross-platform paths.

    Args:
        directory: Directory path to create

    Returns:
        str: Normalized directory path
    """
    normalized_dir = normalize_path(directory)
    os.makedirs(normalized_dir, exist_ok=True)
    return normalized_dir


def ensure_extension(filepath: str, extension: str) -> str:
    """
    Ensure file has the correct extension.

    Args:
        filepath: Original file path
        extension: Required extension (with or without dot)

    Returns:
        str: File path with correct extension
    """
    if not extension.startswith('.'):
        extension = f'.{extension}'

    path_obj = Path(filepath)
    if path_obj.suffix.lower() != extension.lower():
        return str(path_obj.with_suffix(extension))
    return filepath


def get_temp_path(base_dir: str, filename: str) -> str:
    """
    Generate temporary file path in base directory.

    Args:
        base_dir: Base directory for temp files
        filename: Filename for temp file

    Returns:
        str: Full path to temp file
    """
    return safe_join(base_dir, filename)


def clean_filename(filename: str) -> str:
    """
    Clean filename to be safe for all platforms.

    Args:
        filename: Original filename

    Returns:
        str: Cleaned filename
    """
    # Remove problematic characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')

    # Remove multiple consecutive underscores
    while '__' in filename:
        filename = filename.replace('__', '_')

    return filename.strip('_')


def get_relative_path(path: str, base_path: str) -> str:
    """
    Get relative path from base path.

    Args:
        path: Full path
        base_path: Base path to make relative to

    Returns:
        str: Relative path
    """
    try:
        return os.path.relpath(path, base_path)
    except ValueError:
        # Different drives on Windows
        return path


class PathBuilder:
    """Builder class for constructing complex paths safely."""

    def __init__(self, base_path: str):
        self.base_path = normalize_path(base_path)
        self.parts: List[str] = []

    def add(self, *parts: str) -> 'PathBuilder':
        """Add path parts."""
        self.parts.extend(parts)
        return self

    def build(self) -> str:
        """Build the final path."""
        return safe_join(self.base_path, *self.parts)

    def create(self) -> str:
        """Build the path and create directories."""
        path = self.build()
        if not os.path.basename(path).count('.'):  # It's a directory
            create_directory_safely(path)
        else:  # It's a file, create parent directory
            create_directory_safely(os.path.dirname(path))
        return path


# Common path patterns
def get_user_session_path(output_root: str, user_id: str, session_id: str) -> str:
    """Get normalized user session directory path."""
    return PathBuilder(output_root).add(user_id, session_id).create()


def get_audio_file_path(base_dir: str, filename: str = "audio.wav") -> str:
    """Get normalized audio file path."""
    return PathBuilder(base_dir).add(filename).build()


def get_chunks_dir_path(base_dir: str) -> str:
    """Get normalized chunks directory path."""
    return PathBuilder(base_dir).add("chunks").create()


def get_debug_dir_path(base_dir: str, debug_name: str = "debug") -> str:
    """Get normalized debug directory path."""
    return PathBuilder(base_dir).add(debug_name).create()