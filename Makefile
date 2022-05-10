SHELL=/bin/sh

PACAKGES = \
	numpy \
	pandas \
	matplotlib \
	seaborn \
	tqdm \
	kornia[x] \
	kornia_moons
	

TORCH = torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113

DEV_DEPENDENCIES = \
	black \
	mypy \
	isort \
	flake8 \
	jupyterlab \
	jedi-language-server 


install_packages:
	python -m pip install --upgrade pip \
	&& pip install ${TORCH} \
	&& pip install ${DEV_DEPENDENCIES} \
	&& pip install ${PACAKGES}

setup: ## setup package on kaggle docker image
	python --version \
	pip install -r requirements.txt

pip_export:
	pip freeze > requirements.txt

update_datasets:
	zip -r output/sub.zip output/sub
	kaggle datasets version -p ./output/sub -m "Updated data"
