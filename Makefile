# Build the manuscript from the saved experiment outputs.
#
#   make paper      regenerate macros + figures, then compile paper/main.pdf
#   make theory     re-run every simulation experiment (slow: ~40 min)
#   make data       re-collect Binance and HOSE trade records (network)
#   make empirics   re-run the duration analyses on the collected data
#   make all        data -> theory -> empirics -> paper
#   make clean      remove build artefacts (keeps results/ and data/)

PY      := /opt/miniconda3/envs/py313/bin/python
EXP     := experiments
PYPATH  := PYTHONPATH=src:$(EXP)
RUN     := cd $(EXP) && PYTHONPATH=../src:. $(PY) -u

.PHONY: all data theory empirics paper clean distclean

all: data theory empirics paper

data:
	$(PY) data/fetch_binance.py
	$(PY) data/fetch_hose.py

theory:
	$(RUN) exp01_marginal_selfsim.py
	$(RUN) exp02_atom.py
	$(RUN) exp03_potential.py
	$(RUN) exp04_evolution.py
	$(RUN) exp05_correlation.py
	$(RUN) exp06_rho_exact.py

empirics:
	$(RUN) exp10_crypto_durations.py
	$(RUN) exp11_hose_durations.py

paper:
	$(RUN) make_tables.py
	$(RUN) make_figures.py
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

clean:
	cd paper && latexmk -C
	rm -f paper/build.log

distclean: clean
	rm -f paper/results_macros.tex paper/tab_*.tex figs/*.pdf
