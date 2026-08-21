try:
    from .metaworld import MetaWorldEnv
except ImportError:
    MetaWorldEnv = None

try:
    from .robotwin2 import RoboTwin2EnvManager
except Exception:
    RoboTwin2EnvManager = None

try:
    from .adroit import AdroitEnv
except ImportError:
    AdroitEnv = None

try:
    from .peg_assembly import PegAssemblyEnv, PrivilegedPegExpert
except ImportError:
    PegAssemblyEnv = None
    PrivilegedPegExpert = None

try:
    from .dexart import DexArtEnv
except ImportError:
    DexArtEnv = None
