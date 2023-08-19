FROM jupyter/scipy-notebook:python-3.9

ENV LANGUAGE en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV LC_CTYPE en_US.UTF-8
ENV LC_MESSAGES en_US.UTF-8

USER root

# Copy the current directory contents into the container at /app
COPY requirements.txt ./

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8888 available for the app to bind to
EXPOSE 8888

ENV JUPYTER_ENABLE_LAB=yes
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
