dev:
		python preprocess.py

install:
		python -m venv .vc
		. .vc/bin/activate
		pip install azureml-core pandas Pillow nltk wheel torch torchvision