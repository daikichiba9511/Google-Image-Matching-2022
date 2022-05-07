SHELL=/bin/sh

PACAKGES = \
	black \
	mypy \
	isort \
	flake8 \
	python-box \
	bbox-utility \
	pytorch-lightning \
	timm \
	torchmetrics

install_packages:
	python -m pip install --upgrade pip \
	&& python -m pip install ${PACAKGES}

setup: ## setup package on kaggle docker image
	python --version \
	pip install -r requirements.txt

pip_export:
	pip3 freeze > requirements.txt

update_datasets:
	zip -r output/sub.zip output/sub
	kaggle datasets version -p ./output/sub -m "Updated data"
