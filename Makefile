# Convenience targets. Everything runs inside the pinned dolfinx container, since dolfinx,
# PETSc and MPI cannot be installed from PyPI.
#
# The container runs as your own user by default, so files it writes into the mounted tree
# belong to you. Override with `make test USER_FLAG=` to run as root.
IMAGE     ?= fenicsx-ns-dev
USER_FLAG ?= -u $(shell id -u):$(shell id -g)
RUN       := docker run --rm $(USER_FLAG) -v "$(PWD)":/work -w /work $(IMAGE)

.PHONY: help image shell test test-slow test-mpi lint profile docs examples clean fix-permissions

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

# NP is overridable: `make profile NP=4`. Step count and output directory come from
# examples/aorta/Ao11mmrest.yaml, like every other parameter.
NP ?= 8
profile:                 ## PETSc -log_view profile of the aorta case
	docker run --rm $(USER_FLAG) -v "$(PWD)":/work -w /work \
	  -e PETSC_OPTIONS="-log_view :output/profile.txt" $(IMAGE) \
	  mpirun -n $(NP) python3 examples/aorta/run.py
	@echo "wrote output/profile.txt -- look at the fxns_* events and the fxns_time_loop stage"

docs:                    ## build the documentation site
	$(RUN) mkdocs build --strict

# Shortened copies of the shipped configurations, so the smoke run is a smoke run.
# Everything else about each case is whatever its own configuration file says.
define smoke
	@mkdir -p output
	$(RUN) python3 examples/shorten_config.py $(1) output/smoke_$(2).yaml \
	  --max-steps $(3) --output output/smoke_$(2)
	$(RUN) python3 examples/$(2)/run.py --config output/smoke_$(2).yaml
endef

examples:                ## run every example for a few steps as a smoke check
	$(call smoke,examples/turek/turek2d.yaml,turek,20)
	$(call smoke,examples/tube3d/tube3d.yaml,tube3d,20)
	$(call smoke,examples/aorta/Ao11mmrest.yaml,aorta,3)

fix-permissions:         ## reclaim files a root container left behind
	docker run --rm -v "$(PWD)":/work $(IMAGE) chown -R $(shell id -u):$(shell id -g) /work

clean:
	rm -rf output site .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
