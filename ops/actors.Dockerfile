FROM node:22-bookworm-slim
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY tsconfig.server.json ./
COPY proto ./proto
COPY src ./src
RUN npm run build:actors
USER node
CMD ["node", "dist-server/server/actor-coordinator.js"]
