#! /bin/bash

source ../../allen/bin/activate
i=0
while read line
do
    MODEL[ $i ]="$line"        
    (( i++ ))
done < <(ls "../model/allen/"*.py)

readarray -t QUESTION <questions.txt

echo -n MODEL $'\t'>results.tsv

for questiontype in "${QUESTION[@]}"
	do
    	echo -n $questiontype $'\t'>> results.tsv
    done
echo >>	results.tsv

for modeltype in "${MODEL[@]}"
do
	echo -n $modeltype $'\t'>>results.tsv
	for questiontype in "${QUESTION[@]}"
	do
		#echo $questiontype, $modeltype
		answer=$(python3 $modeltype "$questiontype")
		echo -n $answer $'\t'>>results.tsv
	done
	echo >>	results.tsv
done
deactivate



source ../../tfenv/bin/activate
i=0
while read line
do
    T_MODEL[ $i ]="$line"        
    (( i++ ))
done < <(ls "../model/transformers/"*.py)

for modeltype in "${T_MODEL[@]}"
do
	echo -n	$modeltype $'\t'>>results.tsv
	for questiontype in "${QUESTION[@]}"
	do
		#echo $questiontype, $modeltype
		answer=$(python3 $modeltype "$questiontype")
		echo -n	$answer $'\t'>>results.tsv
	done
	echo >>	results.tsv
done
deactivate




