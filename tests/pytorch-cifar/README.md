# RaspAI Capstone



## Setup

### To load:
```
python -m venv .env && source .env/bin/activate && pip install -r requirements.txt
```

### To save if you have used pip, etc:
```
pip freeze > requirements.txt
```



## Setup Steps

1. Download & Install Anaconda [from here](https://github.com/d2deco/31261-raspai-capstone.git)
2. Keep updating,
    1. In Anaconda Navigator once it opens, update it. v2.4.3 minimum.
    2. In a terminal, run `conda update conda`, then supply a `y`
3. To open the shared anaconda environment, run `conda create -n parkpi -f explicit-env.txt`
    - (When done, you need to re-export the anaconda environment. Do so with `conda list -n parkpi --explicit > explicit-env.txt`)
5. Then, get into the env: `conda activate parkpi`
