FROM tensorflow/tensorflow:2.0.0-py3-jupyter

RUN apt-get update && apt-get install -y libmagickwand-dev git && apt-get clean

WORKDIR /workspace
RUN git clone https://github.com/andreacos/BoostingCNN-Jpeg-Primary-Quantization-Matrix-Estimation.git
# Not sure that this requirements is correct (there are another one in 'resources' folder)
# Removing some errorprone dependencies (tf, keras, cv2)
RUN sed -i '1d;4d;10d' BoostingCNN-Jpeg-Primary-Quantization-Matrix-Estimation/requirements.txt
RUN pip install --no-cache-dir numpy==1.19.5 opencv-python-headless==4.0.1.24 &&\
    pip install --no-cache-dir -r BoostingCNN-Jpeg-Primary-Quantization-Matrix-Estimation/requirements.txt

CMD ["bash"]