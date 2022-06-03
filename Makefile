SHELL=/bin/sh

PACAKGES = \
	numpy \
	pandas \
	matplotlib \
	seaborn \
	tqdm \
	kornia[x] \
	kornia_moons \
	opencv-python
	

TORCH = torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113

DEV_DEPENDENCIES = \
	black \
	mypy \
	isort \
	flake8 \
	jupyterlab \
	jedi-language-server 

DETECTRON2 = python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

install_packages:
	python -m pip install --upgrade pip \
	&& python -m pip install --upgrade ${TORCH} \
	&& python -m pip install --upgrade ${DEV_DEPENDENCIES} \
	&& python -m pip install --upgrade ${PACAKGES}

install_detectron2:
	${DETECTRON2}

setup: ## setup package on kaggle docker image
	python --version \
	pip install -r requirements.txt

pip_export:
	pip freeze > requirements.txt

update_datasets:
	zip -r output/sub.zip output/sub
	kaggle datasets version -p ./output/sub -m "Updated data"
