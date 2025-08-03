import os
from lukhed_basic_utils import osCommon as osC

def configure_nfl_stadiums_resource_dir():
    """
    Sets up and monkey patches the nfl_stadiums resource path to ensure it works
    in Lambda (/tmp), Docker, or local environments.
    
    Returns:
        str: Path to the configured resource directory
    """
    resource_dir = os.getenv("NFL_STADIUM_RESOURCES", "/tmp/nfl_stadium_resources")
    os.makedirs(resource_dir, exist_ok=True)

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

    return resource_dir
