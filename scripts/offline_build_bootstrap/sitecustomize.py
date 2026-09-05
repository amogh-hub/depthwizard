"""Install DepthWizard's non-loopback egress guard in PyInstaller hook subprocesses."""

from depthwizard.network_guard import install_strict_offline_network_guard

install_strict_offline_network_guard()
