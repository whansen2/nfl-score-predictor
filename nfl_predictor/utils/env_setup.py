import os

def configure_nfl_stadiums_resource_dir():
    """
    Sets up the nfl_stadiums resource path to ensure it works
    in Lambda (/tmp), Docker, or local environments.
    """
    resource_dir = os.getenv("NFL_STADIUM_RESOURCES", "/tmp/nfl_stadium_resources")
    os.makedirs(resource_dir, exist_ok=True)
    return resource_dir
