# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.245.0/containers/python-3/.devcontainer/base.Dockerfile
 
# [Choice] Python version (use -bullseye variants on local arm64/Apple Silicon): 3, 3.10, 3.9, 3.8, 3.7, 3.6, 3-bullseye, 3.10-bullseye, 3.9-bullseye, 3.8-bullseye, 3.7-bullseye, 3.6-bullseye, 3-buster, 3.10-buster, 3.9-buster, 3.8-buster, 3.7-buster, 3.6-buster
FROM tensorflow/tensorflow:2.15.0-gpu
RUN pip install numpy scipy scikit-image SimpleITK bio-volumentations
RUN pip install importlib-metadata
RUN pip install importlib-resources
RUN pip install json5
RUN pip install jsonref
RUN pip install albumentations
RUN pip install jsonschema
RUN pip install jsonschema-specifications
RUN pip install keras
RUN pip install Keras-Preprocessing
# RUN pip install lifelines
RUN pip install matplotlib
RUN pip install matplotlib-inline
RUN pip install neptune
RUN pip install neptune-tensorflow-keras
RUN pip install nibabel
# RUN pip install oauthlib
RUN pip install OpenEXR
RUN pip install openpyxl
RUN pip install pandas
RUN pip install Pillow
RUN pip install plotly
RUN pip install protobuf
RUN pip install pydicom
RUN pip install pyvista
RUN pip install pylibjpeg
RUN pip install pylibjpeg-libjpeg
RUN pip install pylibjpeg-openjpeg
RUN pip install rasterio
RUN pip install scikit-image
RUN pip install scikit-learn
RUN pip install seaborn
# RUN pip install statsmodels
RUN pip install tensorflow-addons
RUN pip install tensorflow-datasets
RUN pip install tensorflow-estimator
RUN pip install tensorflow-graphics
RUN pip install tensorflow-io
RUN pip install tensorflow-io-gcs-filesystem
RUN pip install tensorflow-metadata
RUN pip install tqdm
# RUN pip install tslearn
# RUN pip install volumentations
RUN pip install ipykernel
RUN pip install opencv-python-headless

# RUN pip install scikit-posthocs
# RUN pip install statannotations


# # Create non-root user.
# ARG USERNAME=tina
# ARG USER_UID=1001
# ARG USER_GID=$USER_UID
 
# RUN groupadd --gid $USER_GID $USERNAME && \
#     useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
#     # Add user to sudoers.
#     apt-get update && \
#     apt-get install -y sudo && \
#     echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
#     chmod 0440 /etc/sudoers.d/$USERNAME && \
#     # Change default shell to bash.
#     usermod --shell /bin/bash $USERNAME