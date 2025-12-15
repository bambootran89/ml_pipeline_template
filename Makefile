SHELL = /bin/bash
PYTHON := python
VENV_NAME = mlproject_env
MAIN_FOLDER = mlproject
TEST_FOLDER = tests
# Docker variables
IMAGE_NAME = ml-pipeline-template
TAG = latest
# Environment
venv:
	${PYTHON} -m venv ${VENV_NAME} && \
	source ${VENV_NAME}/bin/activate && \
	${PYTHON} -m pip install pip setuptools wheel && \
	${PYTHON} -m pip install pip autoflake autopep8 isort flake8 mypy && \
	${PYTHON} -m pip install -e .[dev] && \
	pre-commit install

# Style
style:
	black ./${MAIN_FOLDER}/ --line-length 88
	${PYTHON} -m isort -rc ./${MAIN_FOLDER}/
	${PYTHON} -m  autoflake --in-place --remove-unused-variables --remove-all-unused-imports -r mlproject/
	${PYTHON} -m  autopep8 --in-place --aggressive --aggressive --aggressive -r mlproject/
	flake8 ./${MAIN_FOLDER}/

test:
	${PYTHON} -m flake8 ./${MAIN_FOLDER}/
	${PYTHON} -m mypy ./${MAIN_FOLDER}/
	CUDA_VISIBLE_DEVICES=""  ${PYTHON} -m pytest -v -s --durations=0 --disable-warnings ${TEST_FOLDER}/
	${PYTHON} -m pylint ./${MAIN_FOLDER}/

docker-build:
	docker build -t $(IMAGE_NAME):$(TAG) .

docker-run-api:
	docker run -p 8000:8000 --name ml-api --rm $(IMAGE_NAME):$(TAG)

docker-run-train:
	# Ví dụ chạy training job bằng docker container
	docker run --rm $(IMAGE_NAME):$(TAG) python -m mlproject.src.pipeline.run_pipeline train --config mlproject/configs/experiments/etth1.yaml
