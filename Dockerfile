# peak3d with its browser GUI. From the folder that holds your data:
#   docker run --rm -p 127.0.0.1:8765:8765 -v "$PWD":/data ghcr.io/fanzhoukong/peak3d
# and open the link printed in the log. The command line works too:
#   docker run --rm -v "$PWD":/data ghcr.io/fanzhoukong/peak3d process --input /data/mzml --output /data/out
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/matplotlib

COPY pyproject.toml README.md LICENSE /tmp/src/
COPY peak3d /tmp/src/peak3d
RUN pip install /tmp/src && rm -rf /tmp/src && peak3d --version

LABEL org.opencontainers.image.source="https://github.com/FanzhouKong/project_hoffman" \
      org.opencontainers.image.description="peak3d: LC-MS feature detection with a browser GUI" \
      org.opencontainers.image.licenses="MIT"
WORKDIR /data
EXPOSE 8765
ENTRYPOINT ["peak3d"]
CMD ["gui", "--host", "0.0.0.0", "--port", "8765", "--root", "/data", "--no-browser"]
