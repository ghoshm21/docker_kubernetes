# from the base image
FROM python:3.8.2
# who is maintainer
LABEL maintainer="ghoshm21@gmail.com"
# copy the application data to docker under app dir
COPY . /app
# chage the dir
WORKDIR /app
# upgrade the pip3
RUN apt-get -y update
RUN pip3 install --upgrade setuptools pip
# install all the packages
RUN pip3 install -r requirements.txt
# alter process
# RUN python3 -m pip -r requirements.txt
RUN apt install wget
RUN apt-get -y install openjdk-11-jdk
# which port flask api going to use
EXPOSE 50001
# # start the app
ENTRYPOINT ["python3", "./app.py"]
