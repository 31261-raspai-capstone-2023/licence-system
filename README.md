# licence-system

## Dev Setup
Python version 3.11
```bash
python -m venv venv
```

```bash
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
pip install -r requirements-dev.txt
```

## Prod Setup
### Running the stack via Docker (preferred)
If your looking to spin up the entire stack (web app included), use the docker compose file

First build image for the model server
```bash
docker build . -t licence_system/licence_system:latest
```

Then build the image for the web app from https://github.com/31261-raspai-capstone-2023/web-app
```bash
docker build . -t licence_system/web_app:latest
```

Then spin the stack up
```bash
docker compose up -d
```

### Running standalone via Docker
Build the image
```bash
docker build . -t licence_system/licence_system:latest
```

Run the container
```bash
docker run -d licence_system/licence_system:latest
```

### Running with Python
First install tesseract with your favourite package manager such as
```bash
sudo apt-get install tesseract-ocr
```

Then get the path of tesseract and save this to a variable
```bash
export TESSERACT_CMD="$(which tesseract)"
```

Then install all requirements
```bash
pip install -r requirements.txt
```

Then run the main code 
```bash
python -m licence_system.main
```

## Training
This can either be done through the refactored module code
```bash
python -m licence_system.trainer
```

Or using the Jupyter notebook **lpr-nn-eren.ipynb**

## Inferening
```bash
python -m licence_system.inference
```
