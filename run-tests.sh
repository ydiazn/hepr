#!/bin/bash

coverage run -m unittest discover 
coverage report -m
coverage html