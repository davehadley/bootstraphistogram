FROM  python:3.8.13-bullseye

LABEL org.opencontainers.image.authors="David Hadley <davehadley@users.noreply.github.com>"
LABEL org.opencontainers.image.source="git@github.com:davehadley/bootstraphistogram.git"
LABEL uk.co.davehadley.bootstraphistogram="true"

RUN apt-get update \
    && apt-get -y install sudo \
    && rm -rf /var/lib/apt/lists/* \
    && addgroup bootstraphistogram \
    && adduser --shell /bin/bash --ingroup bootstraphistogram bootstraphistogram \
    && adduser bootstraphistogram sudo \
    && passwd -d bootstraphistogram

USER bootstraphistogram

WORKDIR /home/bootstraphistogram/bootstraphistogram
RUN sudo chown bootstraphistogram:bootstraphistogram /home/bootstraphistogram/bootstraphistogram

RUN python -m pip install --upgrade pip \
    && python -m pip install poetry pre-commit \
    && pre-commit install


SHELL ["/bin/bash", "-c"]

COPY --chown=bootstraphistogram:bootstraphistogram . ./

CMD ["/bin/bash"]

ENV PATH="/home/bootstraphistogram/bootstraphistogram/bin:/home/bootstraphistogram/.local/bin:${PATH}"