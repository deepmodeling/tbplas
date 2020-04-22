#ifdef MKL
# define BLAS
# define VML
# define LAPACK
# define FFTW
# define MKL_DIRECT_CALL
# include <mkl_direct_call.fi>
#endif
