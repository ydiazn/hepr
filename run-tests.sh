#!/bin/bash
source venv/bin/activate

coverage run -m unittest discover 
coverage report -m
coverage html