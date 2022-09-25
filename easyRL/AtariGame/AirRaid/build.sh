#!/usr/bin/env bash

container_name=$1
tag=$2

build(){
  docker build -t "${container_name}":"${tag}" .
}

if [ -e ForCommit ]; then
    rm -rf ForCommit
fi


if git clone git@github.com:lilililidada/ForCommit.git;
then
  cd ./ForCommit || exit
  build
fi

