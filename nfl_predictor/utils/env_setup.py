import os
import shutil
import logging
from typing import Optional
from lukhed_basic_utils import osCommon as osC

logger = logging.getLogger(__name__)

# Global flag to ensure monkey patching only happens once
_MONKEY_PATCH_APPLIED = False

def configure_nfl_stadiums_resource_dir() -> str:
    """
    Sets up and monkey patches the nfl_stadiums resource path to ensure it works
    in Lambda (/tmp), Docker, or local environments.
    
    This function:
    1. Creates the resource directory if it doesn't exist
    2. Copies cached stadium data from the repository to the resource directory
    3. Applies monkey patches to lukhed_basic_utils (only once)
    
    Returns:
        str: Path to the configured resource directory
        
    Raises:
        OSError: If directory creation or file copying fails
    """
    global _MONKEY_PATCH_APPLIED
    
    resource_dir = os.getenv("NFL_STADIUM_RESOURCES", "/tmp/nfl_stadium_resources")
    
    try:
        os.makedirs(resource_dir, exist_ok=True)
        
        # Copy cached stadium data if available
        _copy_cached_stadium_data(resource_dir)
        
        # Apply monkey patches only once
        if not _MONKEY_PATCH_APPLIED:
            _apply_monkey_patches(resource_dir)
            _MONKEY_PATCH_APPLIED = True
            
    except (OSError, IOError) as e:
        logger.error(f"Failed to configure NFL stadiums resource directory: {e}")
        raise
    
    return resource_dir


def _copy_cached_stadium_data(resource_dir: str) -> None:
    """
    Copy cached stadium data from repository to resource directory.
    
    Args:
        resource_dir: Target directory for cached files
    """
    # Find the repository root more robustly
    stadium_cache_dir = _find_stadium_cache_dir()
    
    if not stadium_cache_dir:
        logger.warning("Stadium cache directory not found - will rely on network fetch")
        return
    
    cache_files = ["currentStadiumSoup.txt", "otherStadiumSoup.txt", "parsedSoup.json"]
    
    for cache_file in cache_files:
        src_path = os.path.join(stadium_cache_dir, cache_file)
        dst_path = os.path.join(resource_dir, cache_file)
        
        if not os.path.exists(src_path):
            continue
            
        try:
            # Always copy parsedSoup.json to ensure we have correct data
            # For others, only copy if they don't exist
            if cache_file == "parsedSoup.json" or not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
                logger.debug(f"Copied {cache_file} to resource directory")
        except (OSError, IOError) as e:
            logger.warning(f"Failed to copy {cache_file}: {e}")


def _find_stadium_cache_dir() -> Optional[str]:
    """
    Find the stadium cache directory by walking up the directory tree.
    
    Returns:
        Path to stadium cache directory, or None if not found
    """
    # Start from current file location and walk up
    current_path = os.path.dirname(__file__)
    
    # Walk up the directory tree looking for nfl_stadium_resources
    for _ in range(5):  # Limit search depth
        potential_cache_dir = os.path.join(current_path, "nfl_stadium_resources")
        if os.path.exists(potential_cache_dir):
            return potential_cache_dir
        current_path = os.path.dirname(current_path)
    
    return None


def _apply_monkey_patches(resource_dir: str) -> None:
    """
    Apply monkey patches to lukhed_basic_utils functions.
    
    Args:
        resource_dir: Directory path to return from patched functions
    """
    def patched_create_file_path_string(*args, **kwargs):
        return resource_dir

    def patched_check_create_dir_structure(*args, **kwargs):
        return resource_dir

    def patched_create_dir(*args, **kwargs):
        pass

    # Apply the monkey patches
    osC.create_file_path_string = patched_create_file_path_string
    osC.check_create_dir_structure = patched_check_create_dir_structure
    osC.create_dir = patched_create_dir
    
    logger.debug("Applied monkey patches to lukhed_basic_utils")
