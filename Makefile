.PHONY: make
make:
	python -m pip install -r requirements.txt
	python src/pyclassify/_distance_numba.py
	python -m pip install -e .

# Clean up by uninstalling the package
.PHONY: clean
clean:
	python -m pip uninstall pyclassify -y
