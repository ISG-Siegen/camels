FROM python:3.9.13 as app

RUN apt-get update && apt-get install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev \
    libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev liblzma-dev lzma swig python-dev -y &&  \
    pip install --no-cache-dir --upgrade pip

# Supported libraries
RUN pip install lenskit==0.14.2
RUN pip install scikit-surprise==1.1.2
RUN pip install myfm==0.3.5
RUN pip install xgboost==1.6.2
RUN pip install scikit-learn==1.1.2

# CaMeLS requirements
RUN pip install pandas==1.4.4
RUN pip install numpy==1.22.4
RUN pip install scipy==1.9.1
RUN pip install requests==2.28.1
RUN pip install Flask==2.2.2
RUN pip install xlrd==2.0.1
RUN pip install openpyxl==3.0.10
RUN pip install tables==3.7.0
RUN pip install tabulate==0.8.10
RUN pip install pymfe==0.4.1

