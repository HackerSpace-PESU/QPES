



i=0

readarray -t QUESTION <questions.txt

echo -n MODEL $'\t'>ELMo.tsv

for questiontype in "${QUESTION[@]}"
	do
    	echo -n $questiontype $'\t'>> ELMo.tsv
    done
echo >>	ELMo.tsv

cd ../model/ELMo
pyenv local oldtf
while read line
do
    E_MODEL[ $i ]="$line"        
    (( i++ ))
done < <(ls *.py)
for modeltype in "${E_MODEL[@]}"
do
	echo -n $modeltype\t>>ELMo.tsv
	for questiontype in "${QUESTION[@]}"
	do
		#echo $questiontype, $modeltype
		answer=$(python3 $modeltype "$questiontype")
		echo -n	$answer $'\t'>>ELMo.tsv
	done
	echo >>	ELMo.tsv
done



