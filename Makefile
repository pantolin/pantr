.PHONY: help test coverage clean

help:
	@echo "Commands:"
	@echo "  test      : run the test suite."
	@echo "  coverage  : generate a coverage report."
	@echo "  clean     : remove build artifacts."

# Run the test suite with Numba JIT enabled
test:
	pytest

# Generate an XML coverage report with Numba JIT disabled
coverage:
	NUMBA_DISABLE_JIT=1 pytest --cov=src/pantr --cov-report=xml

# Remove build artifacts
clean:
	rm -rf .pytest_cache .coverage coverage.xml htmlcov/
