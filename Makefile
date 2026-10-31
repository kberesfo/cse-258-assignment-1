# CSE 258 Assignment 1
#   make predict   run every implemented task -> dist/task_N_predictions.csv
#   make bundle    build dist/assignment1.py from src/assignment_1/
#   make check     run the bundled file in a clean temp dir; confirm it reproduces dist/ exactly
#   make submit    predict + bundle + check + writeup.txt -> everything to upload is in dist/
#   make lint      ruff

RUN      ?= uv run
DATA_DIR ?= data
DIST     := dist
CHECKDIR := $(DIST)/.check

.PHONY: predict bundle check submit lint clean

predict:
	$(RUN) assignment-1 --data-dir $(DATA_DIR) --out-dir $(DIST)

bundle:
	$(RUN) python scripts/bundle.py $(DIST)/assignment1.py

check: predict bundle
	rm -rf $(CHECKDIR) && mkdir -p $(CHECKDIR)
	cp $(DIST)/assignment1.py $(CHECKDIR)/
	cd $(CHECKDIR) && $(RUN) python -I assignment1.py $(abspath $(DATA_DIR)) .
	@for f in $(CHECKDIR)/task_*_predictions.csv; do \
		cmp -s $$f $(DIST)/$$(basename $$f) \
			&& echo "check: $$(basename $$f) matches" \
			|| { echo "check: $$(basename $$f) DIFFERS from package output"; exit 1; }; \
	done
	rm -rf $(CHECKDIR)

submit: check
	cp writeup.md $(DIST)/writeup.txt
	@echo "Upload from $(DIST)/:" && ls -1 $(DIST)

lint:
	$(RUN) ruff check src scripts

clean:
	rm -rf $(DIST)/*
