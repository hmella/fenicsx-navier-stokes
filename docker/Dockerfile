# Development and CI image for fenicsx-navier-stokes.
#
# The image provides the ENVIRONMENT only. The project source is not copied in; it is
# mounted at /work when the container runs, so edits on the host take effect immediately
# and the image never carries a stale copy.
#
#   docker build -t fenicsx-ns-dev -f docker/Dockerfile .
#   docker run --rm -it -u "$(id -u):$(id -g)" -v "$PWD":/work fenicsx-ns-dev
#
# See README.md for the full set of commands, and pass -u as above unless you want the
# files the container writes to be owned by root.
#
# The base tag is pinned deliberately. The solver uses the kind="nest" block assembly API
# and V.element.interpolation_points as a property; neither exists in the 0.8 series still
# shipped as dolfinx/dolfinx:stable, so the code will not even import there.
FROM dolfinx/dolfinx:v0.10.0

LABEL org.opencontainers.image.source="https://github.com/hmella/fenicsx-navier-stokes"
LABEL org.opencontainers.image.description="Stabilized incompressible Navier-Stokes in FEniCSx for vascular haemodynamics"
LABEL org.opencontainers.image.licenses="MIT"

# gmsh links against libGL and a few X libraries even when meshing headless.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 libxrender1 libxcursor1 libxft2 libxinerama1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies. Copied on its own so that editing project files never invalidates
# this layer, which is by far the slowest to rebuild.
COPY docker/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# The source is mounted rather than installed, so put it on the import path directly. This
# also means the container works when run as an arbitrary user, which a pip install into
# system site-packages would not.
#
# Prepend rather than assign: the base image already uses PYTHONPATH to expose dolfinx and
# PETSc, and overwriting it makes `import dolfinx` fail.
ENV PYTHONPATH=/work/src:/usr/local/dolfinx-real/lib/python3.12/dist-packages:/usr/local/lib

# Do not write .pyc files. The container may run as root, and root-owned __pycache__
# directories inside the mounted tree cannot be removed by the host user afterwards.
ENV PYTHONDONTWRITEBYTECODE=1

# One thread per MPI rank. The parallelism here is MPI, and the runs are sized to fill the
# machine with ranks; if a threaded BLAS or a threaded hypre also spawned a thread per core
# then 8 ranks on 8 cores would ask for 64 threads and every timing would be meaningless.
# The base image already pins OPENBLAS_NUM_THREADS, but not the generic OpenMP variable.
ENV OMP_NUM_THREADS=1

# FFCx compiles UFL forms to C and caches the result; matplotlib and gmsh also want a
# writable config directory. Point all of them at world-writable locations so the image
# works under any UID, and pre-create them so the first run does not race.
ENV XDG_CACHE_HOME=/tmp/cache \
    MPLCONFIGDIR=/tmp/cache/matplotlib
RUN mkdir -p /tmp/cache/matplotlib && chmod -R 777 /tmp/cache

# Fail the build rather than the first run if anything is missing.
RUN python3 -c "import dolfinx, basix, ufl, petsc4py, mpi4py, gmsh, yaml, pint, scipy, numpy; \
print('dolfinx', dolfinx.__version__, '| basix', basix.__version__, '| ufl', ufl.__version__)"

WORKDIR /work
CMD ["/bin/bash"]
