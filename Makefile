.PHONY: venv, cose_path

venv:
	python3.9 -m venv .venv
	.venv/bin/python -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116 
	.venv/bin/python -m pip install -r requirements.txt
	.venv/bin/python -m pip uninstall mmcv
	.venv/bin/mim install mmcv
	.venv/bin/python -m pip install --editable .

cose_path:
	@echo "COSE_PATH=$$PWD" >> .env
