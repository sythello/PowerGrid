"""Local web interface for the PowerGrid rules engine."""

from .server import PowerGridWebController, make_server

__all__ = ["PowerGridWebController", "make_server"]
