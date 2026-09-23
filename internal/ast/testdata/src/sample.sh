#!/bin/bash
set -e
DIR=/tmp
echo "start"

cleanup() {
  rm -rf "$DIR/$1"
}

cleanup "$@"
