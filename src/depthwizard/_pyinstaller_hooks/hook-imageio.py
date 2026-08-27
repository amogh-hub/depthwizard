from PyInstaller.utils.hooks import copy_metadata

# imageio resolves __version__ with importlib.metadata.version("imageio") at import time. PyInstaller
# collects the Python package through DA3's parallel utility import, but distribution metadata is not
# code and therefore is not included automatically. Preserve the installed distribution metadata so
# the frozen DA3 public API imports exactly as it does in the validated source environment.
datas = copy_metadata("imageio")
