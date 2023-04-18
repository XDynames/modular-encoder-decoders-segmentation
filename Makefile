build:
ifneq ($(wildcard $(CONDA_ENV_PATH)),)
	@echo "Environment $(CONDA_ENV_PATH) already exists, do you want to delete it and create a new one? [y/n]"
	@read ans; \
	if [ $$ans = 'y' ]; then \
		conda env remove -p $(CONDA_ENV_PATH); \
	fi
endif
	# Create conda environment and install packages
	conda create -y -p $(CONDA_ENV_PATH) opencv numpy pyyaml pillow python=3.9 pytorch-lightning \
		black flake8-black flake8 isort loguru pytorch torchvision torchaudio pytorch-cuda=11.7 \
		lightning -c pytorch -c nvidia -c conda-forge	
	
	$(CONDA_ACTIVATE) $(CONDA_ENV_PATH)
	# Install packages using pip
	pip3 install -e .
	