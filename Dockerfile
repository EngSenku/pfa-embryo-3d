# Hugging Face Space image: the web form + the pipeline runner.
#
# The Space itself is light — it fills in parameters, pushes notebooks to the
# researcher's Kaggle account with the Kaggle API and watches Hugging Face for
# the artifacts.  The notebooks bootstrap their own environment on the Kaggle
# side, so this image needs no GPU and no scientific stack.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/hf-cache \
    PFA_RUNS_DIR=/data/pfa-runs

# /data is the persistent mount provided by Hugging Face Spaces (Pro/Team);
# without it the paths simply fall back to the container filesystem.
RUN mkdir -p /data/pfa-runs /data/hf-cache

WORKDIR /app

COPY space/requirements.txt /app/space/requirements.txt
RUN pip install -r /app/space/requirements.txt

# Project code + notebooks (the runner executes these via papermill / Kaggle).
COPY pipeline        /app/pipeline
COPY tools           /app/tools
COPY tests           /app/tests
COPY specs           /app/specs
COPY space           /app/space
COPY run_pipeline.py /app/run_pipeline.py
COPY index5.html     /app/index5.html
COPY *.ipynb         /app/

# Spaces run the container as a non-root user, so make the scratch/cache dirs
# writable by anyone (kept last so the pip layer above stays cacheable).
RUN chmod -R 777 /data

EXPOSE 7860
# Hugging Face Spaces expects 7860; other hosts (Render, Koyeb, Cloud Run, …)
# inject $PORT — honour it with 7860 as the fallback.
CMD ["sh", "-c", "uvicorn space.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
