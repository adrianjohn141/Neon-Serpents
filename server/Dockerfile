FROM node:22-bookworm-slim AS node-runtime

FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY --from=node-runtime /usr/local/bin/node /usr/local/bin/node
COPY --from=node-runtime /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 \
    && pip install -r /tmp/requirements.txt
COPY package.json package-lock.json tsconfig.server.json ./
RUN npm ci
COPY proto ./proto
COPY src ./src
COPY server ./server
RUN mkdir -p server/generated \
    && npm run build:actors \
    && python -m grpc_tools.protoc -I proto --python_out=server/generated --grpc_python_out=server/generated proto/training.proto
EXPOSE 8000 50051
CMD ["python", "-m", "uvicorn", "server.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
