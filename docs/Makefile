PYPATH=$(shell pwd)/../
GIGL_VERSION:=$(shell PYTHONPATH=${PYPATH} \
   python -c 'import giglib as gigl; print(gigl.__version__.split("+")[0])')

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
