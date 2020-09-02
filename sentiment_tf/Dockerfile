# from the base image
FROM python:3.8.3-buster
# who is maintainer
LABEL maintainer="ghoshm21@gmail.com"
# copy the application data to docker under app dir
COPY . /app
# chage the dir
WORKDIR /app
# upgrade the pip3
RUN pip3 install --upgrade setuptools pip
# install all the packages
RUN pip3 install -r requirements.txt
# which port flask api going to use
EXPOSE 5000
# start the app
ENTRYPOINT ["python3", "./app.py"]
