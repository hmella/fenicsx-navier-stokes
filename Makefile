# Convenience targets. Everything runs inside the pinned dolfinx container, since dolfinx,
# PETSc and MPI cannot be installed from PyPI.
#
# The container runs as your own user by default, so files it writes into the mounted tree
# belong to you. Override with `make test USER_FLAG=` to run as root.
IMAGE     ?= fenicsx-ns-dev
USER_FLAG ?= -u $(shell id -u):$(shell id -g)
RUN       := docker run --rm $(USER_FLAG) -v "$(PWD)":/work -w /work $(IMAGE)

.PHONY: help image shell test test-slow test-mpi lint docs examples clean fix-permissions

help:                    ## list the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

image:                   ## build the development container
	docker build -t $(IMAGE) -f docker/Dockerfile .

shell: image             ## interactive shell with the repo mounted at /work
	docker run --rm -it $(USER_FLAG) -v "$(PWD)":/work -w /work $(IMAGE) bash

test:                    ## fast suite (the pull-request tier, about a minute)
	$(RUN) python3 -m pytest tests -q

test-slow:               ## convergence studies and benchmarks
	$(RUN) python3 -m pytest tests -q -m slow

test-mpi:                ## two-rank smoke (the image ships MPICH, so no extra flags)
	$(RUN) mpirun -n 2 python3 -m pytest tests/unit -q -m "not mpi and not slow and not benchmark"

lint:                    ## what CI enforces
	$(RUN) ruff check src examples tests

docs:                    ## build the documentation site
	$(RUN) mkdocs build --strict

examples:                ## run every example at smoke size
	$(RUN) python3 examples/turek/run.py  --case 2d3 --dt 0.02 --max-steps 20 --quiet
	$(RUN) python3 examples/tube3d/run.py --radius 0.5 --length 5 --resolution 0.15 --max-steps 20 --quiet
	$(RUN) python3 examples/aorta/run.py  --max-steps 3 --quiet

fix-permissions:         ## reclaim files a root container left behind
	docker run --rm -v "$(PWD)":/work $(IMAGE) chown -R $(shell id -u):$(shell id -g) /work

clean:
	rm -rf output site .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
