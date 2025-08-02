import os
from lukhed_basic_utils import osCommon as osC

def configure_nfl_stadiums_resource_dir():
    """
    Sets up and monkey patches the nfl_stadiums resource path to ensure it works
    in Lambda (/tmp), Docker, or local environments.
    """
    resource_dir = os.getenv("NFL_STADIUM_RESOURCES", "/tmp/nfl_stadium_resources")
    os.makedirs(resource_dir, exist_ok=True)

    def patched_create_file_path_string(*args, **kwargs):
        return resource_dir

    osC.create_file_path_string = patched_create_file_path_string

    return resource_dir
