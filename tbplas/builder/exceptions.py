"""Exception classes used through the code."""


class CoordLenError(Exception):
    """Exception for coordinate of wrong length."""
    def __init__(self, coord):
        super().__init__()
        self.coord = coord

    def __str__(self):
        return f"length of coordinate {self.coord} not in (2, 3)"


class OrbPositionLenError(CoordLenError):
    """Exception for atomic coordinate of wrong length."""
    def __init__(self, coord):
        super().__init__(coord)

    def __str__(self):
        return f"length of orbital position {self.coord} not in (2, 3)"


class CellIndexLenError(CoordLenError):
    """Exception for cell index of wrong length."""
    def __init__(self, coord):
        super().__init__(coord)

    def __str__(self):
        return f"length of cell index {self.coord} not in (2, 3)"


class LatVecError(Exception):
    """Exception for illegal lattice vectors."""
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "illegal lattice vectors"


class LockError(Exception):
    """Base class for all exceptions of modifying a locked object."""
    def __init__(self):
        super().__init__()
        self.object_name = "object"

    def __str__(self):
        return f"trying to modify a locked {self.object_name}"


class PCLockError(LockError):
    """Exception for modifying a locked primitive cell."""
    def __init__(self):
        super().__init__()
        self.object_name = "primitive cell"


class PCOrbIndexError(Exception):
    """Exception for reading or modifying an orbital with wrong index."""
    def __init__(self, orb_i):
        super().__init__()
        self.orb_i = orb_i

    def __str__(self):
        return f"orbital index {self.orb_i} out of range"


class PCHopDiagonalError(Exception):
    """Exception for treating a diagonal term as hopping term."""
    def __init__(self, rn, orb_i):
        super().__init__()
        self.hop_ind = rn + (orb_i, orb_i)

    def __str__(self):
        return f"hopping term {self.hop_ind} is diagonal"


class PCHopNotFoundError(Exception):
    """Exception for a non-existing Hopping term."""
    def __init__(self, hop_ind):
        super().__init__()
        self.hop_ind = hop_ind

    def __str__(self):
        return f"hopping term {self.hop_ind} not found"


class SCDimSizeError(Exception):
    """Exception for supercell dimension with wrong size."""
    def __init__(self, i_dim, dim_min):
        super().__init__()
        self.i_dim = i_dim
        self.dim_min = dim_min

    def __str__(self):
        return f"dimension on direction {self.i_dim} " \
               f"should be no less than {self.dim_min}"


class SCDimLenError(CoordLenError):
    """Exception for supercell dimension of wrong length."""
    def __init__(self, coord):
        super().__init__(coord)

    def __str__(self):
        return f"length of supercell dimension {self.coord} not in (2, 3)"


class PBCLenError(CoordLenError):
    """Exception for periodic conditions of wrong length."""
    def __init__(self, coord):
        super().__init__(coord)

    def __str__(self):
        return f"length of pbc {self.coord} not in (2, 3)"


class IDPCError(Exception):
    """
    Base class for exceptions associated with index in primitive cell
    representation.
    """
    def __init__(self, id_pc):
        super().__init__()
        self.id_pc = id_pc

    def __str__(self):
        return f"illegal id_pc {self.id_pc}"


class IDPCIndexError(IDPCError):
    """Exception for out of range orbital index in PC representation."""
    def __init__(self, i_dim, id_pc):
        super().__init__(id_pc)
        self.i_dim = i_dim

    def __str__(self):
        index_type = "cell" if self.i_dim in range(3) else "orbital"
        return f"{index_type} index {self.id_pc[self.i_dim]} of id_pc" \
               f" {self.id_pc} out of range"


class VacIDPCIndexError(IDPCIndexError):
    """IDPCIndexError for vacancy."""
    pass


class IDPCLenError(IDPCError):
    """
    Exception for orbital index in PC representation with wrong length.

    id_pc is slightly different from orbital positions or cell indices, so we
    do not derive it from CoordLenError.
    """
    def __init__(self, id_pc):
        super().__init__(id_pc)

    def __str__(self):
        return f"length of id_pc {self.id_pc} is not 4"


class VacIDPCLenError(IDPCLenError):
    """IDPCLenError for vacancy."""
    pass


class IDPCTypeError(IDPCError):
    """Exception for wrong type of orbital index in PC representation."""
    def __init__(self, id_pc):
        super().__init__(id_pc)

    def __str__(self):
        return f"illegal type {type(self.id_pc)} of id_pc"


class VacIDPCTypeError(IDPCTypeError):
    """IDPCTypeError for vacancy."""
    pass


class IDPCVacError(IDPCError):
    """
    Exception where the orbital index in PC representation corresponds to
    a vacancy.
    """
    def __init__(self, id_pc):
        super().__init__(id_pc)

    def __str__(self):
        return f"orbital id_pc {self.id_pc} seems to be a vacancy"


class IDSCError(Exception):
    """
    Base class for exceptions associated with index in supercell
    representation.
    """
    def __init__(self, id_sc):
        super().__init__()
        self.id_sc = id_sc

    def __str__(self):
        return f"illegal id_sc {self.id_sc}"


class IDSCIndexError(IDSCError):
    """Exception for out of range orbital index in SC representation."""
    def __init__(self, id_sc):
        super().__init__(id_sc)

    def __str__(self):
        return f"id_sc {self.id_sc} out of range"


class VacIDSCIndexError(IDSCIndexError):
    """IDSCIndexError for vacancy."""
    pass


class SolverError(Exception):
    """Exception for illegal eigen spectrum solver."""
    def __init__(self, solver):
        super().__init__()
        self.solver = solver

    def __str__(self):
        return f"illegal solver {self.solver}"


class BasisError(Exception):
    """Exception for illegal basis function for evaluating DOS."""
    def __init__(self, basis):
        super().__init__()
        self.basis = basis

    def __str__(self):
        return f"illegal basis function {self.basis}"


class OrbSetLockError(LockError):
    """Exception for modifying a locked orbital set."""
    def __init__(self):
        super().__init__()
        self.object_name = "orbital set"


class SCLockError(LockError):
    """Exception for modifying a locked SuperCell instance."""
    def __init__(self):
        super().__init__()
        self.object_name = "supercell object"


class SCOrbIndexError(Exception):
    """Exception for reading or modifying an orbital with wrong index."""
    def __init__(self, orb_i):
        super().__init__()
        self.orb_i = orb_i

    def __str__(self):
        return f"orbital index {self.orb_i} out of range"


class SCHopDiagonalError(Exception):
    """Exception for treating a diagonal term as hopping term."""
    def __init__(self, rn, orb_i):
        super().__init__()
        self.hop_ind = rn + (orb_i, orb_i)

    def __str__(self):
        return f"hopping term {self.hop_ind} is diagonal"


class InterHopLockError(LockError):
    """Exception for modifying a locked SCInterHopping instance."""
    def __init__(self):
        super().__init__()
        self.object_name = "inter-hopping object"


class InterHopVoidError(Exception):
    """Exception for calling get_* methods of a void SCInterHopping instance."""
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "no hopping terms added to SCInterHopping instance"


class SampleError(Exception):
    """Base class for errors associated to Sample instances."""
    pass


class SampleVoidError(SampleError):
    """Exception for creating a void Sample instance."""
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "no components assigned to Sample instance"


class SampleCompError(SampleError):
    """Exception for adding a component of wrong type to a Sample instance."""
    def __init__(self, i_comp):
        super().__init__()
        self.i_comp = i_comp

    def __str__(self):
        return f"component #{self.i_comp} should be instance" \
               f" of SuperCell or SCInterHopping"


class SampleClosureError(SampleError):
    """
    Exception for an SCInterHopping instance whose sc_bra or sc_ket is not
    included in the sample.
    """
    def __init__(self, i_comp, sc_name):
        super().__init__()
        self.i_comp = i_comp
        self.sc_name = sc_name

    def __str__(self):
        return f"{self.sc_name} of inter_hop #{self.i_comp} not included" \
               f" in sample"
