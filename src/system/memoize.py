class Memoize:
    """
    A class decorator that implements memoization for class instances.

    This decorator caches class instances based on constructor arguments,
    ensuring that repeated instantiation with the same parameters returns
    the same cached instance. This is particularly useful for expensive-to-create
    objects like mathematical functions, basis splines, or computational objects.

    Parameters
    ----------
    cls : class
        The class to be memoized. The class must inherit from `Copyable` or
        implement a similar registry mechanism.

    Methods
    -------
    __call__(*args, **kwargs)
        Handles class instantiation with memoization logic.

    Examples
    --------
    >>> @Memoize
    ... class ExpensiveObject:
    ...     registry = {}
    ...     def __init__(self, param):
    ...         self.param = param
    ...         # Expensive initialization here
    ...
    >>> # First call creates new instance
    >>> obj1 = ExpensiveObject(5)
    >>> # Second call with same parameter returns cached instance
    >>> obj2 = ExpensiveObject(5)
    >>> obj1 is obj2
    True

    Notes
    -----
    - For CardinalBSpline classes, the order 'm' is used as cache key
    - For other classes, it uses None as key, and the classes must define their own keys
    - The decorated class must have a `registry` class attribute (dict)
    - Designed to work with the Copyable pattern for proper instance management
    """

    def __init__(self, cls):
        """
        Initialize the memoization decorator.

        Parameters
        ----------
        cls : class
            The class to be memoized.
        """
        self.cls = cls

    def __call__(self, *args, **kwargs):
        """
        Create or retrieve a class instance with memoization.

        If an instance with the same key (based on arguments) already exists
        in the registry, returns the cached instance. Otherwise, creates a
        new instance, stores it in the registry, and returns it.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the class constructor.
        **kwargs : dict
            Keyword arguments passed to the class constructor.

        Returns
        -------
        object
            A cached instance of the decorated class.

        Notes
        -----
        - For CardinalBSpline: uses the first positional argument (order 'm') as key
        - For other classes: uses None as key, and the class must create their own keys
        """

        if self.cls.__name__ == "CardinalBSpline":
            key = args[0] if len(args) else kwargs['m']

        else:
            key = None

        if key not in self.cls.registry:
            self.cls.registry[key] = self.cls(*args, **kwargs)
        return self.cls.registry[key]
