import threading
from abc import ABCMeta


class SingletonMeta(ABCMeta):
    """
    Thread-safe singleton metaclass per name.

    Ensures that for each unique 'name' (or class name by default),
    only one instance of the class is created. Thread-safe to avoid
    race conditions in multi-threaded environments.
    """

    _instance = {}  # Stores instances keyed by name
    _lock = threading.Lock()  # Ensures thread safety

    def __call__(cls, *args, **kwds):
        # Determine unique key: keyword 'name', first positional arg, or class name
        name = kwds.get("name") or (args[0] if args else cls.__name__)

        with cls._lock:
            # Create instance only if not already present
            if name not in cls._instance:
                cls._instance[name] = super().__call__(*args, **kwds)

        # Return the singleton instance for this name
        return cls._instance[name]
