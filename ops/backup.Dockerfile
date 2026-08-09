FROM alpine:3.22
RUN apk add --no-cache restic postgresql17-client curl ca-certificates \
    && curl -fsSL https://dl.min.io/client/mc/release/linux-amd64/mc -o /usr/local/bin/mc \
    && chmod +x /usr/local/bin/mc
ENTRYPOINT ["/bin/sh", "/backup/backup.sh"]
