import numpy as np
import matplotlib.pylab as plt
from src.nurbs.cpoint import ControlPoint
from src.containers import Container
from src.containers import Knotvector


class ControlPolygon(Container):
    """
        A container for control points used in B-spline and NURBS curve definitions.

        This class manages an ordered collection of ControlPoint objects that define
        the control polygon of a B-spline or NURBS curve. It provides methods for
        refinement, point insertion, and visualization of the control polygon.

        Parameters
        ----------
        keys : array_like, optional
            Integer keys for the control points. If None, sequential keys are used.
        ctrl_points : array_like
            Array of ControlPoint objects defining the polygon.
        name : str, optional
            Optional name identifier for the control polygon.

        Attributes
        ----------
        items : ndarray
            Array of ControlPoint objects.
        keys : ndarray
            Array of integer keys for the control points.
        name : str or None
            Optional name identifier.
        nr_items : int
            Number of control points in the polygon (inherited from Container).
        weights : ndarray
            Array of weights from all control points (property).

        Examples
        --------
        >>> # Create control points
        >>> cp1 = ControlPoint([0, 0], w=1)
        >>> cp2 = ControlPoint([1, 2], w=1)
        >>> cp3 = ControlPoint([2, 0], w=1)
        >>>
        >>> # Create control polygon
        >>> polygon = ControlPolygon(ctrl_points=[cp1, cp2, cp3], name="triangle")
        >>> polygon.plot('ro-')  # Plot with red circles and lines

        Notes
        -----
        - Control points are stored in order of increasing parameter value
        - Supports knot refinement algorithms for B-spline degree elevation
        - Provides methods for both Euclidean and homogeneous coordinate representations
        - Used as input for B-spline and NURBS curve evaluation algorithms
        """

    def __init__(self, keys=None, ctrl_points=None, name=None):
        """
        Initialize a ControlPolygon with control points.

        Parameters
        ----------
        keys : array_like, optional
            Integer keys for the control points. If None, sequential keys
            starting from 0 are generated.
        ctrl_points : array_like
            ControlPoint objects defining the polygon vertices.
        name : str, optional
            Optional name identifier for caching and display.
        """
        super().__init__(keys=keys, items=ctrl_points, name=name)

    def __str__(self):
        """
        Return string representation showing control points in order.

        Returns
        -------
        str
            String with arrow-separated control point representations.
        """
        text = ''
        for key in self.keys:
            text += " -> " + str(self[key])
        return text

    def __repr__(self):
        """
        Return detailed representation with class name and all control points.

        Returns
        -------
        str
            Detailed string showing class name and formatted control points.
        """
        text = f"[{self.__class__.__name__}]"

        if self.name is not None:
            text += self.name

        for key in self.keys:
            text += "\n+-> " + repr(self[key])
        return text

    def plot(self, *args, **kwargs):
        """
        Plot the control polygon using matplotlib.

        Extracts x and y coordinates from all control points and plots
        the polygon with lines connecting consecutive points.

        Parameters
        ----------
        *args : tuple
            Additional positional arguments passed to plt.plot.
        **kwargs : dict
            Additional keyword arguments passed to plt.plot.

        Examples
        --------
        >>> polygon.plot('ro-', linewidth=2, markersize=8)  # Red circles with lines
        >>> polygon.plot('b--', alpha=0.5)  # Blue dashed lines
        """
        x = [self.items[i][0] for i in range(self.nr_items)]
        y = [self.items[i][1] for i in range(self.nr_items)]

        plt.plot(x,y, *args, **kwargs)

    def insert_points(self, keys, points, copy=False):
        """
        Insert new control points at specified key positions.

        Parameters
        ----------
        keys : array_like
            Keys where new points should be inserted.
        points : array_like
            ControlPoint objects to insert.
        copy : bool, optional
            If True, returns a modified copy; if False, modifies in-place.

        Returns
        -------
        ControlPolygon
            Control polygon with inserted points. Returns copy if copy=True.

        Notes
        -----
        - Keys are automatically renumbered to maintain sequential order
        - Insertion preserves the topological order of the control polygon
        - Used in knot insertion and refinement algorithms
        """
        mycopy = self.deepcopy() if copy else self

        idx = np.searchsorted(mycopy.keys, keys)
        mycopy.items = np.insert(mycopy.items, idx, points)
        mycopy.keys = np.arange(mycopy.first_index, mycopy.first_index+mycopy.nr_items+1)
        return self

    def refine(self, tc, tf, copy=True):
        """
        Refine the control polygon using knot insertion algorithm.

        Implements the knot insertion algorithm from Cottrell et al. (Isogeometric Analysis)
        to refine the control polygon when the knot vector is refined. This is used for
        h-refinement in isogeometric analysis.

        Parameters
        ----------
        tc : Knotvector
            Initial (coarse) knot vector.
        tf : Knotvector
            Refined (fine) knot vector.
        copy : bool, optional
            If True, returns a refined copy; if False, refines in-place.

        Returns
        -------
        ControlPolygon
            Refined control polygon corresponding to the refined knot vector.

        References
        ----------
        Cottrell, J. A., Hughes, T. J. R., & Bazilevs, Y. (2009).
        Isogeometric Analysis: Toward Integration of CAD and FEA.
        Wiley.

        Notes
        -----
        - Implements Algorithm from Cottrell et al. (knot insertion)
        - Preserves the curve geometry while increasing degrees of freedom
        - Essential for h-refinement in isogeometric analysis
        """
        mycopy = self.deepcopy() if copy else self
        p = tc.degree

        # Find new knots (knots that exist in tf, but not in tc).
        new_knots = np.setdiff1d(tf.items, tc.items, assume_unique=True)

        for u in new_knots: # Process one knot at a time.
            k = tc.find_span(u)  # Return the 'k' for ' u \in [k, k+1)
            tf = tc.insert_knots([u], copy=True) # The original tf gets lost.

            kf = tf.valid_function_keys
            nf = kf.size

            new_keys = np.empty(shape=(nf,), dtype=type(mycopy.keys[0]))
            new_points = np.empty(shape=(nf,), dtype=type(mycopy.items[0]))

            current_item = -1 # Initialize counter
            for i in kf:
                current_item += 1
                new_keys[current_item] = i

                if i <= k - p:
                    new_points[current_item] = ControlPoint.parse(mycopy[i])

                elif k - p + 1 <= i <= k:
                    nom, den = u - tc[i], tc[i + p] - tc[i]
                    a = 0 if den == 0 else nom / den
                    new_points[current_item] = ControlPoint.parse(a * mycopy[i] + (1 - a) * mycopy[i - 1])

                else:
                    new_points[current_item] = ControlPoint.parse(mycopy[i - 1])


            mycopy.keys = new_keys
            mycopy.items = new_points
            tc = tf.deepcopy()

        return mycopy

    @property
    def weights(self):
        """
        Get the weights of all control points.

        Returns
        -------
        ndarray
            Array of weight values (w) from all control points in order.

        Examples
        --------
        >>> polygon.weights
        array([1., 1., 1., 0.5])
        """
        return np.asarray([self.items[i].w for i in range(self.nr_items)])

    def project(self, copy=True):
        """
        Project control points from homogeneous to Euclidean coordinates.

        Performs perspective division on all control points, converting
        from homogeneous coordinates (x*w, y*w, z*w, w) to Euclidean
        coordinates (x, y, z). This is used to visualize the actual
        geometric positions of NURBS control points.

        Parameters
        ----------
        copy : bool, optional
            If True, returns a projected copy; if False, projects in-place.

        Returns
        -------
        ControlPolygon
            Control polygon with points in Euclidean coordinates.

        Notes
        -----
        - Modifies the coordinates of control points: (x, y, z) → (x/w, y/w, z/w)
        - Essential for visualizing NURBS control polygons
        - The weights remain unchanged after projection
        """
        mycopy = self.deepcopy() if copy else self

        for i in range(mycopy.nr_items):
            mycopy.items[i].coordinates[-1] /= mycopy.items[i].w

        return mycopy