#! /bin/bash
readarray -t QUESTION <questions.txt
for questiontype in "${QUESTION[@]}"
	do
    	echo -n $questiontype
    done