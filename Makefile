.PHONY: run figures test all clean

run:
	node scripts/run_experiment.js

figures:
	node scripts/generate_figures.js

test:
	node test/verify_results.js

all: run figures test

clean:
	rm -rf results/*.json results/*.csv results/*.md figures/*.svg figures/*.png
