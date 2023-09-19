FROM python:3.10.12-bookworm

WORKDIR /root/code

RUN pip3 install ipykernel
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install scikit-learn
RUN pip3 install dash
RUN pip3 install dash[testing]
RUN pip3 install dash_bootstrap_components
RUN pip3 install mlflow

COPY ./code /root/code

CMD tail -f /dev/null