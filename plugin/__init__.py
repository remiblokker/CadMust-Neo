__version__ = "0.1.0"

try:
    from .cadmust_neo_action import CadMustNeoAction
    CadMustNeoAction().register()
except ImportError:
    # Running outside KiCad (e.g., unit tests) — pcbnew not available
    pass
