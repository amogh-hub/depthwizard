from PyInstaller.utils.hooks import copy_metadata

# huggingface_hub decides whether safetensors is available using package-distribution metadata.
# PyInstaller can freeze safetensors.torch while omitting the corresponding .dist-info directory,
# which makes huggingface_hub skip its safetensors imports and later fail with a NameError while
# loading model weights. Preserve the installed safetensors distribution metadata explicitly so the
# frozen runtime reports the same dependency availability as the validated source environment.
datas = copy_metadata("safetensors")
