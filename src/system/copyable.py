import copy
from tabulate import tabulate


class Copyable:
    """
       A mixin class providing object copying, caching, and registry functionality.

       This class implements a registry pattern with deep copying capabilities,
       designed for efficient storage and retrieval of expensive-to-compute objects
       such as mathematical expressions or basis functions. It serves as a base
       class for implementing memoization and object caching systems.

       Attributes
       ----------
       registry : dict
           Class-level dictionary that stores cached instances of Copyable subclasses.
           Keys are typically unique identifiers, values are deep copies of instances.

       Methods
       -------
       deepcopy()
           Returns a deep copy of the current instance.

       build_from(other)
           Performs a shallow copy of attributes from another instance.

       retrieve(key)
           Class method to retrieve a cached instance by key.

       store(key, item, debug=False)
           Class method to store an instance in the registry.

       Examples
       --------
       >>> class ExpensiveComputation(Copyable):
       ...     def __init__(self, params):
       ...         self.params = params
       ...         self.result = self._compute()
       ...
       ...     def _compute(self):
       ...         # Expensive computation here
       ...         return sum(self.params)
       ...
       >>> # Store computation result
       >>> comp = ExpensiveComputation([1, 2, 3])
       >>> ExpensiveComputation.store('key123', comp)
       >>>
       >>> # Retrieve later
       >>> cached_comp = ExpensiveComputation.retrieve('key123')
       >>> print(cached_comp.result)

       Notes
       -----
       - The registry uses deep copying to prevent unintended reference sharing
       - The store() method warns if overwriting existing keys
       - Subclasses should implement their own list() method for pretty printing
       - Designed for immutable or copy-safe objects
       """

    registry = dict()

    def deepcopy(self):
        """
        Create a deep copy of the instance.

        Returns
        -------
        Copyable
            A new instance that is a deep copy of the current object.
        """
        return copy.deepcopy(self)

    def build_from(self, other):
        """
        Perform shallow copy of attributes from another instance.

        Intended for cached items where you want to initialize the current
        instance with the attributes of another instance of the same type.

        Parameters
        ----------
        other : Copyable
            Object to copy attributes from. Must be of the same type as current instance.

        Raises
        ------
        AssertionError
            If `other` is not of the same type as the current instance.
        """
        assert type(other) is type(self), f"Copyable.build_from: The source object should be of type {type(self)}."
        self.__dict__.update(other.__dict__)

    @classmethod
    def retrieve(cls, key):
        """
        Retrieve a cached instance from the registry by key.

        Parameters
        ----------
        key : hashable
            The key used to store the instance in the registry.

        Returns
        -------
        Copyable or None
            A deep copy of the cached instance if found, None otherwise.
        """
        return cls.registry[key].deepcopy() if key in cls.registry else None

    @classmethod
    def store(cls, key, item):
        """
        Store an instance in the class registry.

        Parameters
        ----------
        key : hashable
            The key to use for storing the instance.
        item : Copyable
            The instance to store in the registry.

        Notes
        -----
        Prints a warning message if overwriting an existing key in the registry.
        """
        if key in cls.registry: print(f'EXISTING KEY* {key}', end="\n")  # Keep this to detect accidental overwriting.
        cls.registry[key] = item.deepcopy()
