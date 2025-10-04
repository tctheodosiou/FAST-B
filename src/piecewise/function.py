import numpy as np
from src.system import Copyable, NumericValues
from src.piecewise import Polynomial, Branch


class Function(Copyable):
    """
    A piecewise function composed of multiple Branch objects.

    This class represents piecewise functions where each piece (branch) is
    defined by a rational function (Polynomial numerator and denominator)
    over a specific interval. The Function class handles operations across
    multiple branches, including differentiation, integration, and algebraic
    operations while maintaining proper domain handling.

    Parameters
    ----------
    branches : Branch or iterable of Branch
        The branch(es) that define the piecewise function.
    derivative : int, optional
        The derivative order of the function (default: 0).
    name : str, optional
        Optional name identifier for caching and display.
    merge : bool, optional
        Whether to merge overlapping branches during initialization (default: True).

    Attributes
    ----------
    branches : ndarray
        Array of Branch objects defining the piecewise function.
    derivative : int
        Derivative order of the function.
    name : str or None
        Optional name identifier.
    nr_branches : int
        Number of branches in the function.
    breakpoints : ndarray
        Array of all breakpoints from all branches.
    nr_breakpoints : int
        Number of breakpoints.
    global_support : tuple
        The overall domain of the function (min, max).

    Examples
    --------
    >>> # Create a piecewise function with two branches
    >>> b1 = Branch(support=[0, 1], numerator=Polynomial([1, 0]))  # x on [0,1]
    >>> b2 = Branch(support=[1, 2], numerator=Polynomial([2, -1])) # 2-x on [1,2]
    >>> f = Function([b1, b2], name="triangle")
    >>> f(0.5)  # Evaluate at x=0.5
    0.5

    Notes
    -----
    - Each branch is a rational function defined on a specific interval
    - Operations automatically handle domain intersections and breakpoints
    - Supports caching through the Copyable mixin
    - Can represent discontinuous functions
    """

    @classmethod
    def list(cls):
        """
       List all registered subclasses and their cached instances.

       Returns
       -------
       str
           Formatted string showing all subclasses and their registry contents.
       """
        text = ""
        for sub in cls.__subclasses__():
            text += sub.list() + "\n"
        return text

    @classmethod
    def parse(cls, obj):
        """
        Convert various object types to Function instances.

        Provides flexible conversion from numeric types, Polynomial objects,
        Branch objects, and existing Function instances.

        Parameters
        ----------
        obj : Function, NumericValues, Polynomial, Branch
            Object to convert to Function. Can be:
            - Existing Function instance (returned as-is)
            - Numeric value (converted to constant function on (-∞, ∞))
            - Polynomial (converted to function on (-∞, ∞))
            - Branch (converted to single-branch function)

        Returns
        -------
        Function
            Function representation of the input object.

        Raises
        ------
        RuntimeError
            If the object type cannot be converted to Function.
        """
        if isinstance(obj, cls):
            return obj
        elif isinstance(obj, NumericValues):
            return Function(Branch(numerator=Polynomial(obj), support=(-np.inf, np.inf)))
        elif isinstance(obj, Polynomial):
            return Function(Branch(numerator=obj, support=(-np.inf, np.inf)))
        else:
            raise RuntimeError('Function.parse: unknown type ' + type(obj))

    def __init__(self, branches, derivative=0, name=None, merge=True):
        """
        Initialize a piecewise Function.

        Parameters
        ----------
        branches : Branch or iterable of Branch
            The branch(es) defining the piecewise function.
        derivative : int, optional
            Derivative order (default: 0).
        name : str, optional
            Optional name for caching and display.
        merge : bool, optional
            Whether to merge overlapping branches (default: True).

        Notes
        -----
        - Single Branch objects are automatically converted to lists
        - Setting merge=False can improve performance for cached functions
        - Named functions are cached in the registry for reuse
        """
        # If a single Rational is provided, then it is converted to a list.
        if type(branches) == Branch:
            branches = [branches]

        # Assign to self
        self.branches = np.asarray(branches, dtype=Branch)
        self.derivative = derivative
        self.name = name
        if merge: self.merge_branches()

    @property
    def nr_branches(self) -> int:
        """Return the number of branches in the function."""
        return self.branches.size

    @property
    def breakpoints(self):
        """
        Get all breakpoints from all branches.

        Breakpoints are the boundaries between different branches, extracted
        from the supports of all constituent branches. The returned array
        contains unique, sorted breakpoints.

        Returns
        -------
        ndarray
            Sorted array of all breakpoints in the function.
        """
        x = []

        for i in range(self.nr_branches):
            f = self.branches[i]
            x.append(f.support)
        return np.asarray(np.unique(x), dtype=float)

    @property
    def nr_breakpoints(self):
        """Return the number of breakpoints."""
        return self.breakpoints.size

    @property
    def global_support(self):
        """
        Get the overall domain of the function.

        Returns
        -------
        tuple
            (min_x, max_x) representing the entire domain of the function.
        """
        return self.branches[0].support[0], self.branches[-1].support[1]

    def __str__(self):
        """Return string representation showing all branches."""
        text = ''
        for b in self.branches:
            text += "\n" + str(b)
        return text

    def __repr__(self):
        """Return detailed representation with branch information."""
        text = f"[{self.__class__.__name__}]"
        if self.name is not None:
            text += self.name
        text += f" with {self.nr_branches} branches:"
        for i in range(self.nr_branches):
            text += f"\n+-> {repr(self.branches[i])}"
        return text

    def branches_in_interval(self, support):
        """
        Find branches that are defined in the given interval.

        Parameters
        ----------
        support : array_like
            Interval [a, b] to check for branch support.

        Returns
        -------
        ndarray
            Array of branch indices that are defined in the interval.

        Raises
        ------
        AssertionError
            If support does not have exactly 2 elements.
        """
        support = np.asarray(support, dtype=float)
        assert support.size == 2, "BRANCHES IN INTERVAL: Invalid support provided."

        branch_ids = []
        for i in range(self.nr_branches):
            # Check if the domain midpoint is included. Breakpoints make
            # sure of compatibility.
            if (np.isinf(support[0]) and np.isinf(support[1])) \
                    or self.branches[i].includes(np.mean(support).item()): # INFs and NANs mess with mean.
                branch_ids.append(i)

        return np.asarray(branch_ids)

    def __call__(self, x):
        """
        Evaluate the function at given points.

        The function value is computed as the sum of all branch values
        at each point. Branches return 0 outside their domains.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the function.

        Returns
        -------
        ndarray
            Function values at the input points.
        """
        x = np.asarray(x, dtype=float)
        y = np.zeros_like(x)
        for b in self.branches:
            y += b(x)
        return y

    def diff(self, n: int = 1, copy=True, name=None):
        """
        Compute the n-th derivative of the function.

        Parameters
        ----------
        n : int, optional
            Order of derivative (default: 1).
        copy : bool, optional
            If True, returns a modified copy; if False, modifies in-place.
        name : str, optional
            New name for the derivative function.

        Returns
        -------
        Function
            The n-th derivative as a new Function object.
        """
        df = self.deepcopy() if copy else self
        for b in df.branches:
            b = b.diff(n, copy=False)

        df.derivative += n
        return df

    def integral(self, x1, x2, *args, **kwargs):
        """
        Compute the definite integral of the function.

        The integral is computed as the sum of integrals of all branches
        over their intersection with [x1, x2].

        Parameters
        ----------
        x1 : float
            Lower integration limit.
        x2 : float
            Upper integration limit.
        *args : tuple
            Additional arguments passed to branch integral methods.
        **kwargs : dict
            Additional keyword arguments passed to branch integral methods.

        Returns
        -------
        tuple
            (integral_value, error_estimate) from numerical integration.
        """
        I, err = 0., 0.
        for b in self.branches:
            iI, ierr = b.integral(x1, x2, *args, **kwargs)
            I += iI
            err += ierr
        return I, err

    @property
    def extreme(self):
        """
        Find the extreme value (maximum magnitude) of the function.

        Computes the global extreme by checking all branches' extreme values.
        Used primarily for normalization purposes.

        Returns
        -------
        float
            The extreme value with largest absolute magnitude.
        """
        # Extract extreme points.
        extremes = []
        for b in self.branches:
            extremes.append(b.extreme)

        pos_extreme = np.max(np.array(extremes)).item()
        neg_extreme = np.min(np.array(extremes)).item()
        extreme = pos_extreme if pos_extreme > abs(
            neg_extreme) else neg_extreme

        return extreme

    def normalize(self, copy=True):
        """
        Scale the function so its maximum absolute value is 1.0.

        Parameters
        ----------
        copy : bool, optional
            If True, returns a normalized copy; if False, normalizes in-place.

        Returns
        -------
        Function
            Normalized function with maximum absolute value 1.0.
        """
        F = self.deepcopy() if copy else self

        ymax = F.extreme
        for b in F.branches:
            b.numerator /= ymax

        return F

    def merge_branches(self):
        """
        Merge overlapping branches and simplify the function representation.

        Combines branches that share common intervals by adding their
        rational functions. Updates boundary inclusions for continuous
        and discontinuous domains.

        Returns
        -------
        Function
            Self with merged branches (modified in-place).

        Notes
        -----
        - Creates a single branch per definition domain
        - Handles both continuous and discontinuous boundaries
        - Eliminates branches that cancel to zero
        """
        new_branches = []

        for i in range(self.nr_breakpoints - 1):

            x1, x2 = self.breakpoints[i], self.breakpoints[i + 1]
            supported_branch_ids = self.branches_in_interval(support=(x1, x2))
            # Add all supported polynomials
            R = Branch(numerator=Polynomial(0), support=(x1, x2))

            for branch_id in supported_branch_ids:
                R += self.branches[branch_id]

            # Check if elimination occurred.
            if R.numerator.coeffs[0] == 0:
                continue

            new_branches.append(R)

        # Check if the addition of polynomials leads to total elimination.
        if len(new_branches) == 0:
            new_branches.append(
                Branch(numerator=Polynomial(0), support=(0, 1),
                       includes_left_boundary=True,
                       includes_right_boundary=False, name="NULL")
            )

        # Assign to self.
        self.branches = np.asarray(new_branches, dtype=Branch)

        # Update support boundaries. Currently, all are [_,_)
        #   set to [_,_) for continuous domains.
        #   set to [_,_] for discontinuous domains.

        for i in range(1, self.nr_breakpoints - 1):
            x1, x2 = self.breakpoints[i], self.breakpoints[i + 1]
            supported_branch_ids = self.branches_in_interval(support=(x1, x2))
            if len(supported_branch_ids) == 0:
                self.branches[i - 1].includes_right_boundary = True

        # Update last branch anyway.
        self.branches[-1].includes_right_boundary = True

        return self

    # Define standard operators.
    def __add__(self, other):
        """Add another Function or compatible object."""
        other = Function.parse(other)

        # Append branches of Function to self/copy.
        mycopy = self.deepcopy()
        new_function = Function(branches=[*list(mycopy.branches), *list(other.branches)])
        new_function.merge_branches()

        return new_function

    def __radd__(self, other):
        """Right addition for commutative operation support."""
        return self + other

    def __neg__(self):
        """Return the negative of the function."""
        mycopy = self.deepcopy()
        for i in range(mycopy.nr_branches):
            mycopy.branches[i] = -mycopy.branches[i]
        return mycopy

    def __sub__(self, other):
        """Subtract another function or compatible object."""
        other = Function.parse(other)
        return self + (-other)

    def __rsub__(self, other):
        """Right subtraction (other - self)."""
        return (-self) + (other)

    def __mul__(self, other):
        """Multiply by another function or compatible object."""
        other = Function.parse(other)
        mycopy = self.deepcopy()

        # Find breakpoints from both Functions and merge.
        breakpoint_set_1 = mycopy.breakpoints
        breakpoint_set_2 = other.breakpoints
        total_breakpoints = np.unique(
            np.concatenate([breakpoint_set_1, breakpoint_set_2], axis=0)
        )

        # Loop through intervals as defined by total_breakpoints. If there
        # is a branch from both Functions, multiply them; else skip as 0.

        new_branches = []
        for i in range(len(total_breakpoints) - 1):
            x1, x2 = total_breakpoints[i], total_breakpoints[i + 1]
            branch_1 = mycopy.branches_in_interval(support=(x1, x2))
            branch_2 = other.branches_in_interval(support=(x1, x2))

            # branch_1/2 will have either 1 element or none.
            if len(branch_1) == 0 or len(branch_2) == 0:
                continue
            else:
                # Extract the one element of the list.
                branch_1, branch_2 = branch_1[0], branch_2[0]

            R: Branch = mycopy.branches[branch_1] * other.branches[branch_2]
            new_branch = Branch(numerator=R.numerator, denominator=R.denominator, support=(x1, x2))
            new_branches.append(new_branch)

        # Check for elimination.
        if len(new_branches) == 0:
            null_branch = Branch(numerator=Polynomial(0), support=(0, 1))
            return Function([null_branch], name="NULL")

        return Function(new_branches)

    def __rmul__(self, other):
        """Right multiplication for commutative operation support."""
        return self * other

    def updown(self):
        """Return the reciprocal (1/self) of the function."""
        mycopy = self.deepcopy()
        for i in range(mycopy.nr_branches):
            mycopy.branches[i] = mycopy.branches[i].updown()
        return mycopy

    def __truediv__(self, other):
        """Divide by another function or compatible object."""
        other = Function.parse(other)
        mycopy = self.deepcopy()
        return mycopy * other.updown()

    def __rtruediv__(self, other):
        """Right division (other / self)."""
        other = Function.parse(other)
        return self.updown() * other

    def shift(self, h, copy=True, name=None):
        """
        Shift the function horizontally by h units.

        Parameters
        ----------
        h : float
            Shift amount (positive shifts right).
        copy : bool, optional
            If True, returns a shifted copy; if False, shifts in-place.
        name : str, optional
            New name for the shifted function.

        Returns
        -------
        Function
            Shifted function.
        """
        mycopy = self.deepcopy() if copy else self

        for i in range(mycopy.nr_branches):
            mycopy.branches[i] = mycopy.branches[i].shift(h, copy=False)
        if name is not None:
            mycopy.name = name
        return mycopy

    def scale(self, c, copy=True, name=None):
        """
        Scale the function horizontally by factor c.

        Parameters
        ----------
        c : float
            Scaling factor.
        copy : bool, optional
            If True, returns a scaled copy; if False, scales in-place.
        name : str, optional
            New name for the scaled function.

        Returns
        -------
        Function
            Scaled function.
        """
        mycopy = self.deepcopy() if copy else self

        for i in range(mycopy.nr_branches):
            mycopy.branches[i] = mycopy.branches[i].scale(c, copy=False)
        if name is not None:
            mycopy.name = name
        return mycopy