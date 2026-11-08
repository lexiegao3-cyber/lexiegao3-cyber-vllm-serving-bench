CONFIG ?= configs/l4_qwen2.5-1.5b.yaml
OUT    ?= results/l4-qwen2.5-1.5b

.PHONY: test bench verify report all mock

test:            ## unit tests (no server)
	python -m pytest tests/unit -q

bench:           ## real run: launches vLLM twice (cache on / off)
	python -m bench run --config $(CONFIG) --out $(OUT)

verify:          ## correctness checks over the recorded evidence
	python -m bench verify --results $(OUT)

report:          ## tables + charts, spliced into README.md
	python -m bench report --results $(OUT) --readme README.md

all: bench verify report

mock:            ## whole pipeline against the mock server (no GPU)
	python -m bench run --config configs/mock.yaml --out results/mock-ci
	python -m bench verify --results results/mock-ci
	python -m bench report --results results/mock-ci
