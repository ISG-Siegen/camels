FROM python:3.9.7 as app

RUN apt-get update && apt-get install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev \
    libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev liblzma-dev lzma swig python-dev -y &&  \
    pip install --no-cache-dir --upgrade pip

RUN pip install lenskit==0.14.1
RUN pip install scikit-surprise==1.1.1
RUN pip install scikit-learn==1.1.1
RUN pip install requests==2.27.1
RUN pip install Flask==2.1.2
RUN pip install pandas==1.4.2
RUN pip install numpy==1.22.4
RUN pip install scipy==1.8.1
RUN pip install xlrd==2.0.1

