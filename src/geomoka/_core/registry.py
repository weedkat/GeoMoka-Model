import inspect
from collections import defaultdict


class Registry:
    _MODEL_REGISTRY = defaultdict(dict)
    _METHOD_REGISTRY = defaultdict(dict)

    @classmethod
    def _common_prefix_len(cls, a: str, b: str) -> int:
        a_parts = a.split(".")
        b_parts = b.split(".")
        n = 0
        for x, y in zip(a_parts, b_parts):
            if x != y:
                break
            n += 1
        return n

    @classmethod
    def _get_caller_module(cls) -> str | None:
        frame = inspect.currentframe()
        try:
            # build() -> build_model()/build_method() -> caller
            caller = frame.f_back.f_back.f_back
            return caller.f_globals.get("__name__")
        finally:
            del frame

    @classmethod
    def register(cls, name=None, category="model"):
        def decorator(func):
            module_path = func.__module__
            short_name = name or func.__name__
            registry = getattr(cls, f"_{category.upper()}_REGISTRY")
            registry[short_name][module_path] = func
            return func
        return decorator

    @classmethod
    def build(cls, name, category, module=None, **kwargs):
        # Even though we have module info, we still want to support direct lookup for simplicity
        registry = getattr(cls, f"_{category.upper()}_REGISTRY")

        if name not in registry:
            raise ValueError(f"{category.capitalize()} '{name}' not found in registry.")

        candidates = registry[name]  # {module_path: func}

        if len(candidates) == 1:
            func = next(iter(candidates.values()))
            return func(**kwargs)

        caller_module = module or cls._get_caller_module()
        if not caller_module:
            raise ValueError(
                f"Multiple '{name}' registered, but caller module could not be determined."
            )

        best_func = None
        best_score = -1

        for module_path, func in candidates.items():
            score = cls._common_prefix_len(caller_module, module_path)
            if score > best_score:
                best_score = score
                best_func = func

        if best_func is None or best_score == 0:
            available = ", ".join(sorted(candidates))
            raise ValueError(
                f"Ambiguous {category} '{name}' for caller '{caller_module}'. "
                f"Available modules: {available}"
            )

        return best_func(**kwargs)

    @classmethod
    def register_model(cls, name=None):
        return cls.register(name, "model")

    @classmethod
    def register_method(cls, name=None):
        return cls.register(name, "method")

    @classmethod
    def build_model(cls, name, module=None, **kwargs):
        return cls.build(name, "model", module=module, **kwargs)

    @classmethod
    def build_method(cls, name, module=None, **kwargs):
        return cls.build(name, "method", module=module, **kwargs)