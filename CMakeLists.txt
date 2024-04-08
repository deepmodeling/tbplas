cmake_minimum_required(VERSION 3.17...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C Fortran)

# Search for packages and commands
find_package(
  Python
  COMPONENTS Interpreter Development.Module NumPy
  REQUIRED)
find_program(CYTHON "cython" REQUIRED)

#------------------------------ Compiler options ------------------------------#
# Detect compiler vendor
if(${CMAKE_Fortran_COMPILER} MATCHES ".*gfortran.*")
  set(VENDOR "gnu")
elseif(${CMAKE_Fortran_COMPILER} MATCHES ".*ifort.*")
  set(VENDOR "intel")
elseif(${CMAKE_Fortran_COMPILER} MATCHES ".*ifx.*")
  set(VENDOR "intel")
else()
  message(FATAL_ERROR "Unsupported Fortran compiler ${CMAKE_Fortran_COMPILER}")
endif()

# Check for conflicts
if(USE_INDEX64 AND USE_MKL)
  message(FATAL_ERROR "MKL incompatible with 64-bit array index")
endif()

# Add compiler options for 64bit array index
if(USE_INDEX64)
  if(${VENDOR} STREQUAL "intel")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -i8")
  else()
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fdefault-integer-8")
  endif()
endif()

# Add compiler options for MKL
if(USE_MKL)
  if(${VENDOR} STREQUAL "intel")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -DMKL -qmkl=parallel")
  else()
    message(FATAL_ERROR "MKL incompatible with Fortran compiler ${CMAKE_Fortran_COMPILER}")
  endif()
  if ("$ENV{MKLROOT}" STREQUAL "")
    message(FATAL_ERROR "MKLROOT not set")
  endif()
endif()

#----------------------------- Fortran extension ------------------------------#
# Define the fortranobject
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c
          "import numpy.f2py; print(numpy.f2py.get_include())"
  OUTPUT_VARIABLE F2PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
add_library(fortranobject OBJECT "${F2PY_INCLUDE_DIR}/fortranobject.c")
target_link_libraries(fortranobject PUBLIC Python::NumPy)
target_include_directories(fortranobject PUBLIC "${F2PY_INCLUDE_DIR}")
set_property(TARGET fortranobject PROPERTY POSITION_INDEPENDENT_CODE ON)
if(USE_MKL)
  target_link_libraries(fortranobject PUBLIC mkl_rt iomp5 pthread m dl)
  target_include_directories(fortranobject PUBLIC "$ENV{MKLROOT}/include")
  #target_link_directories(fortranobject PUBLIC "$ENV{MKLROOT}/lib")
endif()

# Generate the interface
add_custom_command(
  OUTPUT f2pymodule.c f2py-f2pywrappers.f
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tbplas/fortran/tbpm.f90"
          "${CMAKE_CURRENT_SOURCE_DIR}/tbplas/fortran/analysis.f90"
          "${CMAKE_CURRENT_SOURCE_DIR}/tbplas/fortran/lindhard.f90"
  VERBATIM
  COMMAND "${Python_EXECUTABLE}" -m numpy.f2py
          "${CMAKE_CURRENT_SOURCE_DIR}/tbplas/fortran/tbpm.f90"
          "${CMAKE_CURRENT_SOURCE_DIR}/tbplas/fortran/analysis.f90"
          "${CMAKE_CURRENT_SOURCE_DIR}/tbplas/fortran/lindhard.f90" 
          -m f2py --lower)

# Define the python module
python_add_library(f2py MODULE
                   "${CMAKE_CURRENT_BINARY_DIR}/f2pymodule.c"
                   "${CMAKE_CURRENT_SOURCE_DIR}/tbplas/fortran/const.f90"
                   "${CMAKE_CURRENT_SOURCE_DIR}/tbplas/fortran/math.F90"
                   "${CMAKE_CURRENT_SOURCE_DIR}/tbplas/fortran/csr.F90"
                   "${CMAKE_CURRENT_SOURCE_DIR}/tbplas/fortran/fft.F90"
                   "${CMAKE_CURRENT_SOURCE_DIR}/tbplas/fortran/random.f90"
                   "${CMAKE_CURRENT_SOURCE_DIR}/tbplas/fortran/propagation.f90"
                   "${CMAKE_CURRENT_SOURCE_DIR}/tbplas/fortran/kpm.f90"
                   "${CMAKE_CURRENT_SOURCE_DIR}/tbplas/fortran/funcs.f90"
                   "${CMAKE_CURRENT_SOURCE_DIR}/tbplas/fortran/tbpm.f90"
                   "${CMAKE_CURRENT_SOURCE_DIR}/tbplas/fortran/analysis.f90"
                   "${CMAKE_CURRENT_SOURCE_DIR}/tbplas/fortran/lindhard.f90"
                   WITH_SOABI)
target_link_libraries(f2py PRIVATE fortranobject)
install(TARGETS f2py DESTINATION ./tbplas/fortran)

#------------------------------ Cython extension ------------------------------#
set(CYX_PREFIX primitive super sample lindhard atom)
foreach(PREF IN LISTS CYX_PREFIX)
  # Generate the interface
  add_custom_command(
    OUTPUT ${PREF}.c
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tbplas/cython/${PREF}.pyx"
    VERBATIM
    COMMAND "${CYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tbplas/cython/${PREF}.pyx"
            --output-file "${CMAKE_CURRENT_BINARY_DIR}/${PREF}.c")

  # Define the python module
  python_add_library(${PREF} MODULE
                    "${CMAKE_CURRENT_BINARY_DIR}/${PREF}.c"
                     WITH_SOABI)
  install(TARGETS ${PREF} DESTINATION ./tbplas/cython)
endforeach()
