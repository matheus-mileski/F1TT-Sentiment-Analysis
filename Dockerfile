FROM jupyter/scipy-notebook:python-3.9

ENV LANGUAGE en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV LC_CTYPE en_US.UTF-8
ENV LC_MESSAGES en_US.UTF-8

USER root

# Add jovyan to the root group
RUN usermod -aG root jovyan

USER jovyan

# Copy the current directory contents into the container at /app
COPY requirements.txt ./
COPY GoogleNews-vectors-negative300.bin.gz ./

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader popular

# Extract GoogleNews-vectors-negative300.bin.gz to word2vec_data and then delete the compressed file
RUN mkdir word2vec_data && \
    gunzip -c GoogleNews-vectors-negative300.bin.gz > word2vec_data/GoogleNews-vectors-negative300.bin && \
    rm GoogleNews-vectors-negative300.bin.gz

# Make port 8888 available for the app to bind to
EXPOSE 8888

ENV JUPYTER_ENABLE_LAB=yes
#CMD [ "python", "-u", "work/notebooks/SentimentAnalysis.py"] # run the script
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

