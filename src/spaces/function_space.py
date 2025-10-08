from src.containers.container import Container


class FunctionSpace(Container):
    """
        A container class for managing collections of basis functions.

        This class represents a function space composed of multiple basis functions,
        typically used in finite element analysis, isogeometric analysis, or other
        numerical methods. It provides organized storage and operations on collections
        of functions with support for differentiation and hierarchical management.

        Parameters
        ----------
        keys : array_like, optional
            Integer keys for the functions. If None, sequential keys are used.
        functions : array_like
            Array of Function objects comprising the function space.
        derivative : int, optional
            The derivative order of the function space (default: 0).
        name : str, optional
            Optional name identifier for the function space.

        Attributes
        ----------
        items : ndarray
            Array of Function objects in the space.
        keys : ndarray
            Array of integer keys for the functions.
        derivative : int
            Derivative order of the function space.
        name : str or None
            Optional name identifier.
        nr_items : int
            Number of functions in the space (inherited from Container).

        Examples
        --------
        >>> # Create a function space with B-spline basis functions
        >>> basis_functions = [bspline1, bspline2, bspline3]
        >>> V_h = FunctionSpace(functions=basis_functions, name="BSpline_Space")
        >>>
        >>> # Differentiate the entire function space
        >>> V_h_prime = V_h.diff(1)
        >>> print(f"Derivative order: {V_h_prime.derivative}")
        1

        Notes
        -----
        - Extends the Container class for efficient function management
        - Maintains consistent derivative order across all functions
        - Supports differentiation of the entire function space
        - Useful for hierarchical function space constructions
        - Commonly used in Galerkin methods and spectral methods
        """

    def __init__(self, keys=None, functions=None, derivative=0, name=None):
        """
        Initialize a FunctionSpace with basis functions.

        Parameters
        ----------
        keys : array_like, optional
            Integer keys for the functions. If None, sequential keys
            starting from 0 are generated.
        functions : array_like
            Function objects that constitute the basis of the space.
        derivative : int, optional
            Initial derivative order of the function space (default: 0).
        name : str, optional
            Optional name identifier for caching and display.
        """
        self.name = name
        self.derivative = derivative
        super().__init__(keys=keys, items=functions, name=name)

    def __str__(self):
        """
        Return compact string representation of the function space.

        Returns
        -------
        str
            Comma-separated list of function names in the space.
        """
        text = ''
        for key in self.keys:
            text += f"{self[key].name}, "
        return text

    def __repr__(self):
        """
        Return detailed representation of the function space.

        Returns
        -------
        str
            Hierarchical representation showing class name and all
            contained functions with their detailed representations.
        """
        text = f"[{self.__class__.__name__}]"

        if self.name is not None:
            text += self.name

        for key in self.keys:
            text += "\n +-> " + repr(self[key])
        return text

    def diff(self, n=1, copy=True):
        """
        Compute the n-th derivative of the entire function space.

        Applies differentiation to all functions in the space and updates
        the derivative order attribute. This is useful for creating
        derivative spaces in variational formulations.

        Parameters
        ----------
        n : int, optional
            Order of derivative to apply (default: 1).
        copy : bool, optional
            If True, returns a differentiated copy; if False, differentiates in-place.

        Returns
        -------
        FunctionSpace
            Function space containing the n-th derivatives of all basis functions.
            Returns copy if copy=True.

        Examples
        --------
        >>> # Create function space and its first derivative
        >>> V = FunctionSpace(functions=basis_functions)
        >>> V_prime = V.diff(1)
        >>> print(f"Original derivative: {V.derivative}")
        0
        >>> print(f"Derivative space: {V_prime.derivative}")
        1

        Notes
        -----
        - Differentiates all functions in the space simultaneously
        - Updates the derivative attribute to track the space order
        - Maintains the same keys and organization as the original space
        - Essential for problems requiring derivative basis functions
        """
        dv = self.deepcopy() if copy else self
        dv.derivative += n

        for f in dv.items:
            f = f.diff(n, copy=False)

        return dv





