# Thin FastAPI wrapper for the OC14 triage model. The vLLM backend runs SEPARATELY (RunPod serverless /
# Modal / a local vLLM process) behind its OpenAI-compatible API; point this wrapper at it via
# VLLM_BASE_URL. The wrapper is small and stateless — run it on cheap always-on infra (or P710).
#
# Build from the REPO ROOT:   docker build -f serving/Dockerfile -t oc14-triage-wrapper .
# Run:   docker run -p 8080:8080 -e VLLM_BASE_URL=https://<endpoint>/v1 -e OC14_MODEL_ID=<model> \
#                    -e OC14_API_KEY=<key> oc14-triage-wrapper
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[serving]"
ENV VLLM_BASE_URL=http://localhost:8000/v1 \
    OC14_MODEL_ID=oc14-triage
EXPOSE 8080
CMD ["uvicorn", "oc14_triage.serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
