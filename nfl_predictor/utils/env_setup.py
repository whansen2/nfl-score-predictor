import os
from lukhed_basic_utils import osCommon as osC

def configure_nfl_stadiums_resource_dir():
    """
    Sets up and monkey patches the nfl_stadiums resource path to ensure it works
    in Lambda (/tmp), Docker, or local environments.
    """
    resource_dir = os.getenv("NFL_STADIUM_RESOURCES", "/tmp/nfl_stadium_resources")
    os.makedirs(resource_dir, exist_ok=True)

    # Store original function
    original_create_dir = osC.create_dir
    
    def patched_create_dir(dirPathList):
        """
        Patched create_dir that redirects any directory creation to our writable resource_dir.
        This prevents nfl_stadiums from trying to create directories in read-only /var/task/
        """
        # If trying to create any nfl_stadium_resources related directories,
        # just ensure our resource_dir exists instead
        if any('nfl_stadium_resources' in str(path) for path in dirPathList):
            os.makedirs(resource_dir, exist_ok=True)
            return
        # For other directories, use original function
        return original_create_dir(dirPathList)

    # Apply the patch
    osC.create_dir = patched_create_dir

    return resource_dir
