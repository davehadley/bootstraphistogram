#!/usr/bin/env bash

if [ ! -z "$BASH" ] && [ ! -z "$BASH_SOURCE" ];
then
    scriptname="${BASH_SOURCE}";
elif  [ ! -z "${ZSH_NAME}" ] && [ ! -z "${(%):-%N}" ];
then
    scriptname="${(%):-%N}";
elif [ ! -z "$KSH_VERSION" ];
then
    scriptname="${.sh.file}"
else
    echo "Unsupported shell detected. Try: bash, zsh or ksh."
    return 1;
fi

SCRIPT_PATH=$(cd -- $(dirname "${scriptname}") && pwd)

if [ ! -d "${SCRIPT_PATH}/venv" ]; 
then
    command -v python3 >/dev/null 2>&1 && PYTHONCMD=python3 || PYTHONCMD=python;
    ${PYTHONCMD} -m venv --prompt "boostraphistogram" ${SCRIPT_PATH}/venv \
    && . ${SCRIPT_PATH}/venv/bin/activate \
    && python -m pip install --upgrade pip \
    && python -m pip install poetry pre-commit
fi
. ${SCRIPT_PATH}/venv/bin/activate

export PATH=${SCRIPT_PATH}/bin:${PATH}

# Install pre-commit hooks if they have not already been installed
if test -d "${SCRIPT_PATH}/.git" && ! test -f "${SCRIPT_PATH}/.git/hooks/pre-commit";
then
    (cd ${SCRIPT_PATH}; pre-commit install)
fi