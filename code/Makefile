VENV := .venv

all: venv

#  sudo apt-get install -y poppler-utils tesseract-ocr

$(VENV)/bin/activate: requirements.txt pyproject.toml
	
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt
	# This one is a pain to install. I don't know how to get this to work by
	# including it in the requirements.txt file. Instead, we install all of our
	# dependencies first, and then install this one separately.
	$(VENV)/bin/pip install -U flash-attn --no-build-isolation

venv: $(VENV)/bin/activate
