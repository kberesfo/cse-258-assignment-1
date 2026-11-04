# CSE 258 Assignment 1
#   make predict   snapshot dist/ to dist_history/<timestamp>/, then run every task -> dist/
#   make changes   rows changed in each prediction file vs the latest snapshot
#   make bundle    build dist/assignment1.py from src/assignment_1/
#   make check     run the bundled file in a clean temp dir; confirm it reproduces dist/ exactly
#   make submit    predict + bundle + check + writeup.txt -> everything to upload is in dist/
#   make lint      ruff

RUN      ?= uv run
DATA_DIR ?= data
DIST     := dist
CHECKDIR := $(DIST)/.check
HIST     := $(DIST)_history

.PHONY: predict bundle check submit lint clean snapshot changes

# Copy the current predictions to dist_history/<timestamp>/ before they are overwritten.
snapshot:
	@if ls $(DIST)/task_*_predictions.csv >/dev/null 2>&1; then \
		d=$(HIST)/$$(date +%Y%m%d-%H%M%S); mkdir -p $$d; \
		cp $(DIST)/task_*_predictions.csv $$d/; echo "snapshot -> $$d"; \
	fi

predict: snapshot
	$(RUN) assignment-1 --data-dir $(DATA_DIR) --out-dir $(DIST)

# Share of prediction rows that changed vs the most recent snapshot.
changes:
	@last=$$(ls -1d $(HIST)/*/ 2>/dev/null | sort | tail -1); \
	if [ -z "$$last" ]; then echo "no snapshot yet"; exit 0; fi; \
	echo "vs $$last"; \
	for f in $(DIST)/task_*_predictions.csv; do \
		p=$$last$$(basename $$f); [ -f $$p ] || continue; \
		awk -v name=$$(basename $$f) 'NR==FNR{a[FNR]=$$0;next} a[FNR]!=$$0{c++} \
			END{printf "  %-26s %6d / %d rows changed (%.2f%%)\n", name, c, FNR, 100*c/FNR}' $$p $$f; \
	done

bundle:
	$(RUN) a1-bundle $(DIST)/assignment1.py

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
