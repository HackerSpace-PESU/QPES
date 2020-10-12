#! /bin/bash

i=0
while read line
do
    MODEL[ $i ]="$line"        
    (( i++ ))
done < <(ls "../model/"*.py)
readarray -t QUESTION <questions.txt

echo -n QUESTION, > results.txt
for modeltype in "${MODEL[@]}"
    do
        echo -n $modeltype, >> results.txt 
    done
echo >> results.txt

for questiontype in "${QUESTION[@]}"
do
    echo -n $questiontype, >> results.txt
    for modeltype in "${MODEL[@]}"
    do
        #echo $questiontype, $modeltype
        answer=$(python3 $modeltype "$questiontype" "$modeltype")
        echo -n $answer, >> results.txt 
    done
    echo >> results.txt
done