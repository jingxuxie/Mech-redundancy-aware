PYTHON ?= python
export PYTHONPATH := src
export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
.PHONY: test unpack figures paper experiments

test:
	$(PYTHON) -m pytest -q

# Optional: available when using the complete downloadable research bundle.
unpack:
	$(PYTHON) experiments/unpack_results.py

figures:
	@if test -f results/combinatorial.csv; then \
	  $(PYTHON) experiments/run_all.py --suite plot && $(PYTHON) experiments/summarize.py; \
	else $(PYTHON) experiments/render_release.py; fi

paper: figures
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex

experiments:
	$(PYTHON) experiments/run_all.py --suite all --seeds 20 --steps 1200
	$(PYTHON) experiments/rank_sweep.py --seeds 20
	$(PYTHON) experiments/verify_theory.py
	$(PYTHON) experiments/stress.py
	$(PYTHON) experiments/summarize.py
