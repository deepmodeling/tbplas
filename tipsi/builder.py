"""builder.py contains functions and classes for sample building.

Functions
----------
    bc_default
        Default boundary conditions.
    grouper
        Make sublist iterables for a list, for parallel computation.
    hop_dict_ft
        Calculate Fourier transform of a Lattice, HopDict pair.
    interpolate_k_points
        Get list of momenta by interpolation between symmetry points.
    band_structure
        Calculate band structure for a Lattice, HopDict pair.
    uniform_strain
        Uniformly strain a Lattice, HopDict pair.
    extend_unit_cell
        Extend the unit cell.

Classes
----------
    HopDict
        Contains relative hopping values.
    Lattice
        Contains lattice information.
    SiteSet
        Contains site tags.
    Sample
        Contains all the information of the tight-binding system:
        geometry, hoppings, boundary conditions, etc.
"""

################
# dependencies
################

# plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.collections as mc
except ImportError:
    print("Plotting functions not available.")

# input & output
try:
    import h5py
except ImportError:
    print("h5py functions not available.")

# multiprocessing
import multiprocessing as mp

# numerics & math
import copy
import itertools
import numpy as np
import numpy.linalg as npla
import scipy.sparse as spsp
import scipy.linalg.lapack as spla

################
# helper functions
################


def bc_default(unit_cell_coords, orbital):
    """Default (closed) boundary conditions.

    Parameters
    ----------
    unit_cell_coords : 3-tuple of integers
        unit cell coordinates
    orbital : integer
        orbital index

    Returns
    ----------
    unit_cell_coords : 3-tuple of integers
        unit cell coordinates
    orbital : integer
        orbital index
    """

    return unit_cell_coords, orbital


def grouper(in_list, n):
    """Make n sublists from list.

    Parameters
    ----------
    in_list : list
        input list
    n : integer
        max number of sublists to return

    Returns
    ----------
    sublists : iterable
        iterable over sublists
    n_sublists : integer
        number of sublists
    """

    n_list = len(in_list)
    in_list = iter(in_list)
    len_sublists = int(np.ceil(1. * n_list / n))
    if (len_sublists * (n - 1) == n_list):
        # in this case the nth sublist is empty, so:
        n_sublists = n - 1
    else:
        n_sublists = n
    sublists = iter(lambda: list(itertools.islice(in_list, len_sublists)), [])
    return sublists, n_sublists


def hop_dict_ft(hop_dict, lattice, momentum):
    """Calculate Fourier transform of a HopDict, Lattice pair.

    Parameters
    ----------
    hop_dict : HopDict object
        contains electronic information
    lattice : Lattice object
        contains geometric information
    momentum : 3-list of floats
        momentum [kx, ky, kz]

    Returns
    -----------
    Hk : (nr_orbitals, nr_orbitals) list of complex floats
        k-space Hamiltonian
    """

    # prepare
    nr_orbitals = len(lattice.orbital_coords)
    sparse_hop_dict = hop_dict.sparse()
    Hk = np.zeros((nr_orbitals, nr_orbitals), dtype=complex)

    # iterate over orbitals
    for orb0 in range(nr_orbitals):
        r0 = lattice.site_pos((0, 0, 0), orb0)
        # iterate over HopDict items
        for tag, hop in sparse_hop_dict[orb0].items():
            # transform and add to Hk
            x, y, z, orb1 = tag
            r1 = lattice.site_pos((x, y, z), orb1)
            dr = np.subtract(r1, r0)
            r_dot_k = np.dot(momentum, dr)
            Hk[orb0, orb1] += np.exp(1j * r_dot_k) * hop

    return Hk


def interpolate_k_points(k_points, resolution):
    """Get list of momenta by interpolation between
    symmetry points.

    Parameters
    ----------
    k_points : (n, 3) list of floats
        k-point coordinates
    resolution : integer
        number of momenta between two k-points

    Returns
    ----------
    momenta : ((n - 1) * resolution + 1, 3) list of floats
        interpolated k-point coordinates
    xvals : ((n - 1) * resolution + 1) list of floats
        x-axis values for band plot
    ticks : (n) list of floats
        list of xvals corresponding to symmetry points
    """

    # get momenta
    momenta = []
    for i in range(len(k_points) - 1):
        kp0 = k_points[i]
        kp1 = k_points[i + 1]
        for j in range(resolution):
            rat = 1. * j / resolution
            momenta.append([kp0[0] + (kp1[0] - kp0[0]) * rat, \
                            kp0[1] + (kp1[1] - kp0[1]) * rat, \
                            kp0[2] + (kp1[2] - kp0[2]) * rat])
    momenta.append(k_points[-1])

    # get xvals
    xvals = [0]
    for i in range(len(momenta) - 1):
        diff = np.subtract(momenta[i + 1], momenta[i])
        xvals.append(xvals[-1] + np.linalg.norm(diff))

    # get ticks
    ticks = [xvals[i * resolution] for i in range(len(k_points))]
    ticks.append(xvals[-1])

    return momenta, xvals, ticks


def band_structure(hop_dict, lattice, momenta):
    """Calculate band structure for a HopDict, Lattice pair.

    Parameters
    ----------
    hop_dict : HopDict object
        contains electronic information
    lattice : Lattice object
        contains geometric information
    momenta : (n_momenta, 3) list of floats
        momenta [kx, ky, kz] for band structure calculation

    Returns
    -----------
    bands : (n_momenta, n_orbitals) list of complex floats
        list of energies corresponding to input momenta
    """

    # prepare
    momenta = np.array(momenta)
    n_momenta = momenta.shape[0]
    n_orbitals = len(lattice.orbital_coords)
    bands = np.zeros((n_momenta, n_orbitals))

    # iterate over momenta
    for i in range(n_momenta):

        # fill k-space Hamiltonian
        momentum = momenta[i, :]
        Hk = hop_dict_ft(hop_dict, lattice, momentum)

        # get eigenvalues, store
        eigenvalues, eigenstates, info = spla.zheev(Hk)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        bands[i, :] = eigenvalues[:]

    return bands


def uniform_strain(lattice_old, hop_dict_old, strain_tensor, beta):
    """Uniformly strain a Lattice, HopDict pair.

    Parameters
    ----------
    lattice_old : tipsi.Lattice
        lattice to strain
    hop_dict_old : tipsi.HopDict
        hopping dictionary to strain
    strain_tensor : (3,3)-list of floats
        strain tensor
    beta : float
        strain coefficient

    Returns
    ----------
    lattice_new : tipsi.Lattice
        strained lattice
    hop_dict_new : tipsi.HopDict
        strained hopping dictionary
    """

    # rescale lattice
    one_plus_eps = np.diag([1., 1., 1.]) + strain_tensor
    vectors_new = [np.dot(one_plus_eps, vector) \
                   for vector in lattice_old.vectors]
    orbital_coords_new = [np.dot(one_plus_eps, coord) \
                          for coord in lattice_old.orbital_coords]
    lattice_new = Lattice(vectors_new, orbital_coords_new)

    # rescale hop_dict
    hop_dict_new = HopDict()
    for uc, hop in hop_dict_old.dict.items():
        hop_dict_new.empty(uc, hop.shape)
        for i in range(hop.shape[0]):
            for j in range(hop.shape[1]):
                hopval = hop[i, j]
                r_old = npla.norm(lattice_old.site_pos(uc, j) \
                                  - lattice_old.site_pos((0,0,0), i))
                r_new = npla.norm(lattice_new.site_pos(uc, j) \
                                  - lattice_new.site_pos((0,0,0), i))
                if r_old != 0.0:
                    hopval_new = hopval * np.exp(-beta * (r_new / r_old - 1.))
                    hop_dict_new.set_element(uc, (i, j), hopval_new)
                else:
                    hop_dict_new.set_element(uc, (i, j), hopval)

    return lattice_new, hop_dict_new


def extend_unit_cell(lattice_old, hop_dict_old, direction, amount):
    """Extend the unit cell in a certain direction. Especially
    helpful for including bias.

    Parameters
    ----------
    lattice_old : tipsi.Lattice
        lattice to extend
    hop_dict_old : tipsi.HopDict
        hopping dictionary to extend
    direction : integer
        index of lattice vector giving direction of extension
    amount : integer
        number of unit cells to combine

    Returns
    ----------
    lattice_new : tipsi.Lattice
        extended lattice
    hop_dict_new : tipsi.HopDict
        extended hopping dictionary
    """

    # get useful parameters
    d = direction
    nr_orb_old = len(lattice_old.orbital_coords)

    # function for transforming to new coordinate system
    def extend_coords(unit_cell_coord, orbital):
        r_old = unit_cell_coord[d]
        r_new = int(np.floor(r_old / amount))
        orb_new = orbital + (r_old - r_new) * nr_orb_old
        uc_new = list(unit_cell_coord)
        uc_new[d] = r_new
        return tuple(uc_new), orb_new

    # extend lattice
    orbital_coords_new = lattice_old.orbital_coords
    for i in range(1, amount):
        orbital_coords_new = np.append(orbital_coords_new, \
            lattice_old.orbital_coords + i * lattice_old.vectors[d], \
            axis = 0)
    vectors_new = lattice_old.vectors.copy()
    vectors_new[d] *= amount
    lattice_new = Lattice(vectors_new, orbital_coords_new)
    lattice_new.extended *= amount

    # extend hop_dict
    hop_dict_new = HopDict()
    for k in range(amount):
        for uc, hop in hop_dict_old.dict.items():
            for i in range(hop.shape[0]):
                for j in range(hop.shape[1]):
                    uc_transpose = list(uc)
                    uc_transpose[d] += k
                    uc_new, orb1 = extend_coords(tuple(uc_transpose), j)
                    if uc_new not in hop_dict_new.dict.keys():
                        hop_dict_new.empty(uc_new, (hop.shape[0] * amount, \
                                                    hop.shape[1] * amount))
                    hopval = hop[i, j]
                    orb0 = i + k * nr_orb_old
                    hop_dict_new.set_element(uc_new, (orb0, orb1), hopval)

    return lattice_new, hop_dict_new


################
# classes
################


class HopDict:
    """HopDict class

    A hopping dictionary contains relative hoppings.

    Attributes
    ----------
    dict : dictionary
        dictionary with site tags as keys and complex matrices as values
    """

    def __init__(self):
        """Initialize hop_dict object
        """

        self.dict = {}

    def set(self, rel_unit_cell, hopping):
        """Add hopping matrix to dictionary.

        Parameters
        ----------
        rel_unit_cell : 3-tuple of integers
            relative unit cell coordinates
        hopping : complex float or matrix of floats
            hopping value or matrix of hopping values
        """

        # turn single number into list
        if (type(hopping) not in [list, np.ndarray]):
            hopping = [[hopping]]

        # add to dictionary
        hopping = np.array(hopping, dtype=complex)
        self.dict[rel_unit_cell] = hopping

    def empty(self, rel_unit_cell, shape):
        """Add empty hopping matrix to dictionary.

        Parameters
        ----------
        rel_unit_cell : 3-tuple of integers
            relative unit cell coordinates
        shape : 2-tuple of integers
            shape of empty hopping matrix
        """

        # add empty matrix to dictionary
        empty_mat = np.zeros(shape, dtype=complex)
        self.dict[rel_unit_cell] = empty_mat

    def set_element(self, rel_unit_cell, element, hop):
        """Add single hopping to hopping matrix.

        Parameters
        ----------
        rel_unit_cell : 3-tuple of integers
            relative unit cell coordinates
        element : 2-tuple of integers
            element indices
        hop : complex float
            hopping value
        """

        self.dict[rel_unit_cell][element[0], element[1]] = hop

    def delete(self, rel_unit_cell):
        """Delete hopping matrix from dictionary.

        Parameters
        ----------
        rel_unit_cell : 3-tuple of integers
            relative unit cell coordinates

        Returns
        -----------
        boolean
            True if site was deleted, false if not
        """

        return self.dict.pop(rel_unit_cell, None)

    def add_conjugates(self):
        """Adds hopping conjugates to self.dict."""

        # declare new dict
        self.new_dict = copy.deepcopy(self.dict)
        # iterate over items
        for rel_unit_cell, hopping in self.dict.items():
            x, y, z = rel_unit_cell
            reverse_unit_cell = (-x, -y, -z)
            # create reverse_unit_cell dict key
            if reverse_unit_cell not in self.dict:
                reverse_hopping = np.conjugate(np.transpose(hopping))
                self.new_dict[reverse_unit_cell] = reverse_hopping
            # then, make sure all conjugate hoppings are there
            for i in range(hopping.shape[0]):
                for j in range(hopping.shape[1]):
                    hop = hopping[i, j]
                    hop_conjg = self.new_dict[reverse_unit_cell][j, i]
                    if (not hop == 0.) and (hop_conjg == 0.):
                        self.new_dict[reverse_unit_cell][j, i] = np.conjugate(
                            hop)
        # done
        self.dict = self.new_dict

    def remove_z_hoppings(self):
        """Remove z-direction hoppings.
        """

        dict_new = {}
        for uc, hop in self.dict.items():
            if uc[2] == 0:
                dict_new[uc] = hop
        self.dict = dict_new

    def sparse(self):
        """Get sparse hopping dictionary.

        Returns
        -----------
        sparse hopping dictionary
        """

        # make sure conjugates are added
        self.add_conjugates()

        # get maximum orbital index
        max_orb = 0
        for rel_unit_cell, hopping in self.dict.items():
            max_orb = np.amax(hopping.shape + (max_orb, ))

        # declare sparse dict
        sparse_dict = [{} for i in range(max_orb)]

        # iterate over elements, add nonzero elements to sparse dict
        for rel_unit_cell, hopping in self.dict.items():
            for i0 in range(hopping.shape[0]):
                for i1 in range(hopping.shape[0]):
                    hop = hopping[i0, i1]
                    if (hop != 0j):
                        sparse_dict[i0][rel_unit_cell + (i1, )] = hop

        return sparse_dict


class Lattice:
    """Lattice class

    A lattice object contains information about the geometry of the lattice.

    Attributes
    -----------
    vectors : (3,3) numpy array
        array of lattice vectors [a_0, a_1, a_2]
    vectorsT : (3,3) numpy array
        transposed lattice vectors
    orbital_coords : (n,3) numpy array
        array of orbital coordinates for all n orbitals
    extended : integer
        number of times the unit cell has been extended, default 1
    """

    def __init__(self, vectors, orbital_coords):
        """Initialize lattice object.

        Parameters
        ----------
        vectors : (2 or 3)-list of float 3-tuples
            list of lattice vectors
        orbital_coords : list of float 3-tuples
            list of orbital positions within unit cell
        """

        self.vectors = np.array(vectors)
        if (len(self.vectors) == 2):
            self.vectors = np.append(self.vectors, [[0., 0., 1.]], axis=0)
        self.vectorsT = np.transpose(self.vectors)
        self.orbital_coords = np.array(orbital_coords)
        self.extended = 1

    def site_pos(self, unit_cell_coords, orbital):
        """Get orbital position in Cartesian coordinates.

        Parameters
        ----------
        unit_cell_coords : integer 3-tuple
            unit cell coordinates
        orbital : integer
            orbital index

        Returns
        -----------
        numpy array of floats
            x, y and z coordinates of orbital
        """

        return self.orbital_coords[orbital] \
               + np.dot(self.vectorsT, unit_cell_coords)

    def area_unit_cell(self):
        """Get unit cell area.

        Returns
        -----------
        float
            unit cell area
        """

        a = self.vectors[0, :]
        b = self.vectors[1, :]
        return npla.norm(np.cross(a, b))

    def volume_unit_cell(self):
        """Get unit cell volume.

        Returns
        -----------
        float
            unit cell volume
        """

        a = self.vectors[0, :]
        b = self.vectors[1, :]
        c = self.vectors[2, :]
        return np.inner(a, np.cross(b, c))

    def reciprocal_latt(self):
        """Get reciprocal lattice vectors.

        Returns
        -----------
        (3,3) numpy array of floats
            array containing [k_0, k_1, k_2]
        """

        vec = self.vectors
        div = np.inner(vec[0], np.cross(vec[1], vec[2]))
        rec = [2.0 * np.pi * np.cross(vec[(i + 1) % 3], vec[(i + 2) % 3]) / div \
               for i in range(3)]
        return np.array(rec)


class SiteSet:
    """SiteSet class

    A SiteSet object contains a dict of site tags.

    Attributes
    ----------
    sites : dict
        dict of site tags in the sample
    """

    def __init__(self):
        """Initialize sample object.
        """

        self.sites = {}

    def add_site(self, unit_cell_coords, orbital=0):
        """Add orbital to sample.

        Parameters
        ----------
        unit_cell_coords : 3-tuple of integers
            unit cell coordinates
        orbital : integer, optional
            orbital index
        """

        # add site tag to self.sites
        self.sites[unit_cell_coords + (orbital, )] = None

    def delete_site(self, unit_cell_coords, orbital):
        """Delete orbital from sample.

        Parameters
        ----------
        unit_cell_coords : 3-tuple of integers
            unit cell coordinates
        orbital : integer
            orbital index
        """

        # delete site tag from self.sites
        self.sites.pop(unit_cell_coords + (orbital, ), None)


class Sample:
    """Sample class

    A Sample object contains sample information, such as sites and hoppings.

    Attributes
    ----------
    lattice : lattice object
        the lattice used for the sample
    bc_func : function
        function for boundary conditions
    nr_processes : int
        number of processes to use for parallel functionality
    rescale : float
        Hamiltonian rescale value; default value: 1.
    index_to_tag : list of 3-tuples
        ordered list of sites in sample
    tag_to_index : dict (keys: 3-tuple, values: integer)
        dictionary giving index for each site tag
    site_x : list of floats
        site x-locations
    site_y : list of floats
        site y-locations
    site_z : list of floats
        site z-locations
    indices : numpy array of integers
        csr-indices for hoppings & distances
    indptr : numpy array of integers
        csr-indptr for hoppings & distances
    hop : numpy array of complex floats
        sparse Hamiltonian
    dx : numpy array of floats
        x-distances of hoppings
    dy : numpy array of floats
        y-distances of hoppings
    """

    def __init__(self, lattice, site_set = [], bc_func = bc_default, \
                 nr_processes = 1, read_from_file = False):
        """Index site_set and store site locations.

        Parameters
        ----------
        lattice : Lattice object
            the lattice to use for the sample
        site_set : SiteSet object, optional
            contains sites to add to the sample
        bc_func : function, optional
            function for boundary conditions
        nr_processes : integer, optional
            number of parallel processes
        read_from_file : string, optional
            Filename, in case we read a Sample from file. Default: False.
        """

        # read from file
        if read_from_file:

            # init input
            self.lattice = lattice
            self.bc_func = bc_func
            self.nr_processes = nr_processes

            # open file
            try:
                f = h5py.File(read_from_file, 'r')
            except:
                print("Cannot find file to read!")
                return

            # read data
            self.rescale = f["sample/rescale"][0]
            self.index_to_tag = [tuple(tag) for tag in \
                                 f["sample/index_to_tag"][:]]
            self.tag_to_index = {tag: i for i, tag in \
                                 enumerate(self.index_to_tag)}
            self.site_x = f["sample/site_x"][:]
            self.site_y = f["sample/site_y"][:]
            self.site_z = f["sample/site_z"][:]
            self.indices = f["sample/indices"][:]
            self.indptr = f["sample/indptr"][:]
            self.hop = f["sample/hop"][:]
            self.dx = f["sample/dx"][:]
            self.dy = f["sample/dy"][:]
            return

        # if not read_from_file, declare attributes
        self.lattice = lattice
        self.bc_func = bc_func
        self.nr_processes = nr_processes
        self.rescale = 1
        self.indices = np.array([])
        self.indptr = np.array([0])
        self.hop = np.array([])
        self.dx = np.array([])
        self.dy = np.array([])

        # indexing
        self.index_to_tag = list(site_set.sites)
        self.index_to_tag.sort(
            key=lambda tag: tag
        )  # this will speed up FORTRAN / use itemgetter() instead?
        self.tag_to_index = {tag: i for i, tag in \
                             enumerate(self.index_to_tag)}

        if self.nr_processes == 1:  # no multiprocessing

            # get site locs
            data = self.__get_locs(self.index_to_tag)
            self.site_x = data[0]
            self.site_y = data[1]
            self.site_z = data[2]

        else:  # use multiprocessing

            # divide up sites list
            sites_div, N = grouper(self.index_to_tag, self.nr_processes)

            # get site locs in parallel
            pipes = [mp.Pipe() for i in range(N)]
            processes = [None for i in range(N)]
            data = [None for i in range(N)]
            for i, tags in enumerate(sites_div):
                pipe = pipes[i]
                processes[i] = mp.Process(target=self.__get_locs, \
                                          args=(tags, pipe[1]))
                processes[i].start()

            # collect results
            scan = [True for i in range(N)]
            while True:
                running = any(scan)
                for i in range(N):
                    pipe = pipes[i]
                    if scan[i] and pipe[0].poll():
                        data[i] = pipe[0].recv()
                        scan[i] = False
                        pipe[0].close()
                if not running:
                    break
            for p in processes:
                p.join()

            # put data in class attribute arrays
            self.site_x = data[0][0]
            self.site_y = data[0][1]
            self.site_z = data[0][2]
            for i in range(1, N):
                self.site_x = np.concatenate((self.site_x, data[i][0]))
                self.site_y = np.concatenate((self.site_y, data[i][1]))
                self.site_z = np.concatenate((self.site_z, data[i][2]))

            # cleanup
            del data

    def get_loc(self, tag):
        return self.lattice.site_pos(tag[0:3], tag[3])

    def __get_locs(self, tags, conn=False):
        """Private method for getting site location of a list of tags.

        Sends site_r through conn Pipe.

        Parameters
        ----------
        tags : list of 4-tuples
            site tags
        conn : multiprocessing.Pipe object
            Pipe through which to send data. If False,
            no pipe is used; data is returned normally"""

        # declare output array
        site_r = np.zeros((3, len(tags)), dtype=float)
        # iterate over tags
        for i, tag in enumerate(tags):
            #site_r[:,i] = self.get_loc(tag)
            site_r[:, i] = self.lattice.site_pos(tag[0:3], tag[3])  # faster
        # return results
        if conn:
            # send results through pipe
            conn.send(site_r)
            conn.close()
            return
        else:
            # return results normally
            return site_r

    def __get_hoppings(self, tags, sparse_hop_dict, conn=False):
        """Private method for getting hopping data for list of tags.

        Sends indices, indptr, hop, dx, dy lists through conn Pipe.

        Parameters
        ----------
        tags : list of 4-tuples
            site tags
        sparse_hop_dict : dict
            sparse hopping dictionary
        conn : multiprocessing.Pipe object
            Pipe through which to send data. If False,
            no pipe is used; data is returned normally."""

        # declare arrays
        indices = []
        indptr = [0]
        hop = []
        dx = []
        dy = []

        # iterate over all sites
        for tag0 in tags:
            r0 = tag0[0:3]
            orb0 = tag0[3]
            i0 = self.tag_to_index[tag0]
            indptr.append(indptr[-1])
            for rel_tag, hopping in sparse_hop_dict[orb0].items():
                tag1 = (r0[0] + rel_tag[0], \
                        r0[1] + rel_tag[1], \
                        r0[2] + rel_tag[2], \
                        rel_tag[3])
                # if tag in sample, add
                try:
                    i1 = self.tag_to_index[tag1]
                    indices.append(i1)
                    indptr[-1] += 1
                    hop.append(hopping)
                    dx.append(self.site_x[i1] - self.site_x[i0])
                    dy.append(self.site_y[i1] - self.site_y[i0])
                # check pbc
                except KeyError:
                    r1 = tag1[0:3]
                    orb1 = tag1[3]
                    pbc_r, pbc_orb = self.bc_func(r1, orb1)
                    pbc_tag = pbc_r + (pbc_orb, )
                    try:
                        i1 = self.tag_to_index[pbc_tag]
                        pos = self.lattice.site_pos(r1, orb1)
                        indices.append(i1)
                        indptr[-1] += 1
                        hop.append(hopping)
                        dx.append(pos[0] - self.site_x[i0])
                        dy.append(pos[1] - self.site_y[i0])
                    except KeyError:
                        pass

        # return results
        if conn:
            # send results through Pipe
            conn.send([
                np.array(indices, dtype=int),
                np.array(indptr, dtype=int),
                np.array(hop, dtype=complex),
                np.array(dx, dtype=float),
                np.array(dy, dtype=float)
            ])
            conn.close()
            return
        else:
            # return results normally
            return (np.array(indices, dtype=int), np.array(indptr, dtype=int),
                    np.array(hop, dtype=complex), np.array(dx, dtype=float),
                    np.array(dy, dtype=float))

    def add_hop_dict(self, hop_dict):
        """Apply hopping dictionary.

        Parameters
        ----------
        hop_dict : hopping dictionary object
            hopping information
        """

        sparse_hop_dict = hop_dict.sparse()

        if self.nr_processes == 1:  # no multiprocessing

            # apply hopping dictionary
            data = self.__get_hoppings(self.index_to_tag, \
                                       sparse_hop_dict)
            self.indices = data[0]
            self.indptr = data[1]
            self.hop = data[2]
            self.dx = data[3]
            self.dy = data[4]

        else:  # use multiprocessing

            # divide up sites list
            sites_div, N = grouper(self.index_to_tag, self.nr_processes)

            # get hopping data in parallel
            pipes = [mp.Pipe() for i in range(N)]
            processes = [None for i in range(N)]
            data = [None for i in range(N)]
            for i, tags in enumerate(sites_div):
                pipe = pipes[i]
                processes[i] = mp.Process(target=self.__get_hoppings, \
                                          args=(tags, sparse_hop_dict, pipe[1]))
                processes[i].start()

            # collect results
            # scan over processes
            scan = [True for i in range(N)]
            while any(scan):
                for i in range(N):
                    pipe = pipes[i]
                    if scan[i] and pipe[0].poll():
                        # get data, close process
                        data[i] = pipe[0].recv()
                        scan[i] = False
                        pipe[0].close()
                        processes[i].join()

            # put data in class attribute arrays
            self.indptr = [0]
            indices = []
            hop = []
            dx = []
            dy = []
            for i in range(N):
                self.indptr = np.concatenate(
                    (self.indptr, np.array(data[i][1][1:]) + self.indptr[-1]))
                indices = itertools.chain(indices, data[i][0])
                hop = itertools.chain(hop, data[i][2])
                dx = itertools.chain(dx, data[i][3])
                dy = itertools.chain(dy, data[i][4])
            del data
            self.indices = np.fromiter(indices, dtype=int, count=-1)
            self.hop = np.fromiter(hop, dtype=complex, count=-1)
            self.dx = np.fromiter(dx, dtype=float, count=-1)
            self.dy = np.fromiter(dy, dtype=float, count=-1)

    def delete_hopping(self, unit_cell_coord0, unit_cell_coord1, \
                       orbital0 = 0, orbital1 = 0):
        """Delete hopping.

        Parameters
        ----------
        unit_cell_coord0 : 3-tuple
            unit cell coordinate for site 0
        unit_cell_coord1 : 3-tuple
            unit cell coordinate for site 1
        orbital0 : int, optional
            orbital 0
        orbital1 : int, optional
            orbital 1

        Returns
        -----------
        bool
            True if hopping value is deleted
            False if hopping value is not found
        """

        # get site tags and indices
        tag0 = unit_cell_coord0 + (orbital0, )
        i0 = self.tag_to_index[tag0]
        tag1 = unit_cell_coord1 + (orbital1, )
        if tag1 in self.tag_to_index:
            i1 = self.tag_to_index[tag1]
        else:
            pbc_r1, pbc_orb1 = self.bc_func(unit_cell_coord1, orbital1)
            pbc_tag = pbc_r1 + (pbc_orb1, )
            if pbc_tag in self.tag_to_index:
                i1 = self.tag_to_index[pbc_tag]

        # check if hopping already exists
        # if yes, delete hopping & distance values
        subindices = self.indices[self.indptr[i0]:self.indptr[i0 + 1]]
        for i, j in enumerate(subindices):
            if j == i1:
                # delete hopping
                self.indices = np.delete(self.indices, self.indptr[i0] + i)
                self.hop = np.delete(self.hop, self.indptr[i0] + i)
                self.dx = np.delete(self.dx, self.indptr[i0] + i)
                self.dy = np.delete(self.dy, self.indptr[i0] + i)
                for i in range(i0 + 1, len(self.indptr)):
                    self.indptr[i] -= 1
                return True

        # if not, there's nothing to delete
        return False

    def set_hopping(self, hop, unit_cell_coord0, unit_cell_coord1, \
                    orbital0 = 0, orbital1 = 0):
        """Add or change hopping, automatically add conjugate.

        Parameters
        ----------
        hop : float
            hopping value
        unit_cell_coord0 : 3-tuple
            unit cell coordinate for site 0
        unit_cell_coord1 : 3-tuple
            unit cell coordinate for site 1
        orbital0 : int, optional
            orbital 0
        orbital1 : int, optional
            orbital 1

        Returns
        -----------
        bool
            True if hopping value is changed
            False if hopping value is added
        """

        # this method needs to be improved
        # move change/insert code to separate methods
        # take care of situation where H is empty when calling this method

        # get site tags, indices and distances
        tag0 = unit_cell_coord0 + (orbital0, )
        i0 = self.tag_to_index[tag0]
        r0 = self.get_loc(tag0)
        tag1 = unit_cell_coord1 + (orbital1, )
        r1 = self.get_loc(tag1)
        if tag1 in self.tag_to_index:
            i1 = self.tag_to_index[tag1]
            r1 = self.get_loc(tag1)
        else:
            pbc_r1, pbc_orb1 = self.bc_func(unit_cell_coord1, orbital1)
            pbc_tag = pbc_r1 + (pbc_orb1, )
            if pbc_tag in self.tag_to_index:
                i1 = self.tag_to_index[pbc_tag]
            else:
                print("Site not in sample.")
        dist_x = r1[0] - r0[0]
        dist_y = r1[1] - r0[1]

        # check if hopping already exists
        # if yes, change hopping value
        subindices = self.indices[self.indptr[i0]:self.indptr[i0 + 1]]
        for i, j in enumerate(subindices):
            if j == i1:
                self.hop[self.indptr[i0] + i] = hop
        # conjugate
        subindices = self.indices[self.indptr[i1]:self.indptr[i1 + 1]]
        for i, j in enumerate(subindices):
            if j == i0:
                self.hop[self.indptr[i1] + i] = hop
                return True

        # if not, add hopping value and distance values
        self.indices = np.insert(self.indices, self.indptr[i0], i1)
        self.hop = np.insert(self.hop, self.indptr[i0], hop)
        self.dx = np.insert(self.dx, self.indptr[i0], dist_x)
        self.dy = np.insert(self.dy, self.indptr[i0], dist_y)
        for i in range(i0 + 1, len(self.indptr)):
            self.indptr[i] += 1
        # conjugate
        self.indices = np.insert(self.indices, self.indptr[i1], i0)
        self.hop = np.insert(self.hop, self.indptr[i1], hop)
        self.dx = np.insert(self.dx, self.indptr[i1], dist_x)
        self.dy = np.insert(self.dy, self.indptr[i1], dist_y)
        for i in range(i1 + 1, len(self.indptr)):
            self.indptr[i] += 1
        return False

    def rescale_H(self, value=False):
        """Rescale Hamiltonian.

        Parameters
        ----------
        value : float, positive, optional
            All hoppings are divided by this value.
            Choose it such that the absolute value
            of the largest eigenenergy is smaller than 1.
            If no value is chosen, a good value is found,
            but this is really slow for large matrices.

        Returns
        ----------
        value : float
            Rescale value.
        """

        # if user doesn't provide rescale value, calculate it
        # this is really show though
        if value == False:
            value = 0.
            for i in range(len(self.indptr) - 1):
                max_val = np.sum([np.absolute(self.hop[self.indptr[i] \
                                                       :self.indptr[i+1]])])
                value = np.amax((max_val, value))

        # store rescale, rescale H
        self.hop /= (value / self.rescale)
        self.rescale = value

        return value

    def energy_range(self):
        """Energy range to consider in calculations.

        Returns
        ----------
        en_range : float
            All eigenvalues are between (-en_range / 2, en_range / 2)
        """

        en_range = 2. * self.rescale
        return en_range

    def plot(self,
             fig_name='system.png',
             single_site_coord=False,
             single_site_orbital=0,
             draw_size=5,
             draw_dpi=600):
        """Plot sample in 2D, save to file.

        Parameters
        ----------
        fig_name : string, optional
            save to this file
        single_site_coord : int 3-tuple int, optional
            if not False, only print hoppings to a single site with this
            site coordinate
        single_site_orbital : int, optional
            only print hoppings to a single site with this orbital index
        draw_size : float, optional
            scale site and hopping drawing size
        draw_dpi : integer, optional
            dpi of image
        """

        fig, ax = plt.subplots()

        # plot sites
        plt.scatter(
            self.site_x,
            self.site_y,
            s=0.5 * draw_size**2,
            c='black',
            zorder=2,
            edgecolors='none')

        # put hoppings in LineCollection
        hops = []
        linews = []
        H = spsp.csr_matrix((self.hop, self.indices, self.indptr))
        H = H.tocoo()
        for i, j, hop in zip(H.row, H.col, H.data):
            if not single_site_coord:
                if i > j:
                    hops.append([[self.site_x[i], self.site_y[i]],
                                 [self.site_x[j], self.site_y[j]]])
                    linews.append(draw_size * npla.norm(hop))
            else:
                site_i_coord = self.index_to_tag[i]
                if site_i_coord == single_site_coord + (single_site_orbital, ):
                    hops.append([[self.site_x[i], self.site_y[i]],
                                 [self.site_x[j], self.site_y[j]]])
                    linews.append(draw_size * npla.norm(hop))
        lines = mc.LineCollection(
            hops, linewidths=linews, colors='grey', zorder=1)

        # plot hoppings
        ax.add_collection(lines)
        ax.set_aspect('equal')
        plt.axis('off')
        plt.draw()
        plt.savefig(fig_name, bbox_inches='tight', dpi=draw_dpi)
        plt.close()

    def save(self, filename="sample.hdf5"):
        """Save sample

        Parameters
        ----------
        filename : string
            Save to this hdf5 file. Default value: "sample.hdf5".
        """

        # save everything to a hdf5 file
        # except lattice, bc_func
        f = h5py.File(filename, 'w')
        grp = f.create_group("sample")
        grp.create_dataset("rescale", data=[self.rescale])
        grp.create_dataset("index_to_tag", data=self.index_to_tag)
        #grp.create_dataset("tag_to_index", data=self.tag_to_index)
        grp.create_dataset("site_x", data=self.site_x)
        grp.create_dataset("site_y", data=self.site_y)
        grp.create_dataset("site_z", data=self.site_z)
        grp.create_dataset("indices", data=self.indices)
        grp.create_dataset("indptr", data=self.indptr)
        grp.create_dataset("hop", data=self.hop)
        grp.create_dataset("dx", data=self.dx)
        grp.create_dataset("dy", data=self.dy)
        f.close()

    def Hk(self, momentum):
        """Calculate the Fourier transform of the Hamiltonian.

        Parameters
        ----------
        momentum : 3-list of floats
            momentum [kx, ky, kz]

        Returns
        -----------
        Hk : (tot_nr_orbitals, tot_nr_orbitals) list of complex floats
            k-space Hamiltonian
        """

        # prepare
        tot_nr_orbitals = len(self.index_to_tag)
        Hk = np.zeros((tot_nr_orbitals, tot_nr_orbitals), \
                      dtype = complex)

        # fill Hk
        for i0 in range(tot_nr_orbitals):
            for i in range(self.indptr[i0], self.indptr[i0 + 1]):
                i1 = self.indices[i]
                dz = self.site_z[i1] - self.site_z[i0]
                dr = [self.dx[i], self.dy[i], dz]
                r_dot_k = np.dot(momentum, dr)
                Hk[i0, i1] += np.exp(1j * r_dot_k) * self.hop[i]

        return Hk

    def band_structure(self, momenta):
        """Calculate band structure of the Sample.

        Parameters
        ----------
        momenta : (n_momenta, 3) list of floats
            momenta [kx, ky, kz] for band structure calculation

        Returns
        -----------
        bands : (n_momenta, n_tot_orbitals) list of complex floats
            list of energies corresponding to input momenta
        """

        # prepare
        momenta = np.array(momenta)
        n_momenta = momenta.shape[0]
        n_tot_orbitals = len(self.index_to_tag)
        bands = np.zeros((n_momenta, n_tot_orbitals))

        # iterate over momenta
        for i in range(n_momenta):

            # fill k-space Hamiltonian
            momentum = momenta[i, :]
            Hk = self.Hk(momentum)

            # get eigenvalues, store
            eigenvalues, eigenstates, info = spla.zheev(Hk)
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            bands[i, :] = eigenvalues[:] * self.rescale

        return bands

    def set_magnetic_field(self, B):
        """Set magnetic field

        Parameters
        ----------
        B : float
            magnetic field in Tesla, distance units must be in
            nanometers, hoppings must be in eV
        """

        # apply Peierls substitution
        n_tot_orbitals = len(self.index_to_tag)
        for i0 in range(n_tot_orbitals):
            for i in range(self.indptr[i0], self.indptr[i0 + 1]):
                i1 = self.indices[i]
                ytot = self.site_y[i0] + self.site_y[i1]
                phase = 1j * np.pi * B * self.dx[i] * ytot / 4135.666734
                self.hop[i] = self.hop[i] * np.exp(phase)
