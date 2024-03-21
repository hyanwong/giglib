# Need to set PYTHONPATH so that we pick up the local tsdate
PYPATH=$(shell pwd)/../
GIGL_VERSION:=$(shell PYTHONPATH=${PYPATH} \
   python3 -c 'import GeneticInheritanceGraphLibrary as gigl; print(gigl.__version__.split("+")[0])')

BUILDDIR      = _build

all: dev

dev:
	PYTHONPATH=${PYPATH} ./build.sh

dist:
	@echo Building distribution for tsdate version ${GIGL_VERSION}
	sed -i -e s/__GIGL_VERSION__/${GIGL_VERSION}/g _config.yml
	PYTHONPATH=${PYPATH} ./build.sh

clean:
	rm -fR $(BUILDDIR)