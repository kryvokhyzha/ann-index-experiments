run_embedding_generator:
	python src/prepare_embeddings.py
run_test:
	python src/benchmarks.py --data-type real --embedding-column embedding && \
	python src/benchmarks.py --data-type real --embedding-column proj128 && \
	python src/benchmarks.py --data-type real --embedding-column proj64 && \
	python src/benchmarks.py --data-type real --embedding-column proj32 && \
	python src/benchmarks.py --data-type real --embedding-column proj16 && \
	python src/benchmarks.py --data-type real --embedding-column proj8 && \
	python src/benchmarks.py --data-type synthetic --embedding-column embedding && \
	python src/benchmarks.py --data-type synthetic --embedding-column proj128 && \
	python src/benchmarks.py --data-type synthetic --embedding-column proj64 && \
	python src/benchmarks.py --data-type synthetic --embedding-column proj32 && \
	python src/benchmarks.py --data-type synthetic --embedding-column proj16 && \
	python src/benchmarks.py --data-type synthetic --embedding-column proj8

poetry_install_deps:
	poetry install --no-root
poetry_get_lock:
	poetry lock
poetry_update_deps:
	poetry update
poetry_update_self:
	poetry self update
poetry_show_deps:
	poetry show
poetry_show_deps_tree:
	poetry show --tree
poetry_build:
	poetry build

download_dataset:
	git clone https://github.com/clinc/oos-eval.git && \
	cp -r oos-eval/data . && \
	rm -rf oos-eval

pre_commit_install: .pre-commit-config.yaml
	pre-commit install
pre_commit_run: .pre-commit-config.yaml
	pre-commit run --all-files
pre_commit_rm_hooks:
	pre-commit --uninstall-hooks

nvsmi0:
	watch -n 0.1 nvidia-smi -i 0
