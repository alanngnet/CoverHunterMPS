PYTHON = python3
PIP = $(PYTHON) -m pip
VIRTUALENV = $(PYTHON) -m venv
VENV_OPTS += --clear
VENV = .venv
in_venv = . $(VENV)/bin/activate &&

$(VENV)/bin/activate:
	$(VIRTUALENV) $(VENV_OPTS) $(VENV)
	$(in_venv) $(PIP) install --upgrade pip setuptools wheel

$(VENV)/bin/coverage: $(VENV)/bin/activate
	$(in_venv) $(PIP) install -r requirements.txt

tests coverage: TESTS = `find tests/ -name 'test_*.py'`

.PHONY: tests
tests: $(VENV)/bin/coverage
	$(in_venv) $(PYTHON) -Wd -m unittest -c $(TESTS)

.PHONY: coverage
coverage: $(VENV)/bin/coverage
	$(in_venv) $(PYTHON) -Wd -m coverage run \
		--source=src \
		-m unittest -c $(TESTS)
	$(in_venv) $(PYTHON) -m coverage report
	$(in_venv) $(PYTHON) -m coverage html

.PHONY: format
format: $(VENV)/bin/coverage
	$(in_venv) black src tests
