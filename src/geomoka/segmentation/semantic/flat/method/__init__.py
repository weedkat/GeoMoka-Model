from geomoka._core.registry import Registry
from .supervised.main import run_supervised
from .unimatch_v2.main import run_unimatch_v2

@Registry.register_method
