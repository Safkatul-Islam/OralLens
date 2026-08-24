FROM node:22-alpine AS build

WORKDIR /opt/orallens/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/index.html frontend/tsconfig.json frontend/tsconfig.node.json frontend/vite.config.ts ./
COPY frontend/src ./src
COPY frontend/public ./public

RUN npm run build

FROM nginx:1.29-alpine

COPY infra/nginx/default.conf /etc/nginx/conf.d/default.conf
COPY infra/nginx/40-runtime-config.sh /docker-entrypoint.d/40-runtime-config.sh
COPY --from=build /opt/orallens/frontend/dist /usr/share/nginx/html

RUN sed -i 's/\r$//' /docker-entrypoint.d/40-runtime-config.sh \
    && chmod 0755 /docker-entrypoint.d/40-runtime-config.sh

EXPOSE 8080
