#!/usr/bin/env bash
cd "/home/Downloads/rawcull-app"
exec .venv/bin/python -m rawcull "$@"
