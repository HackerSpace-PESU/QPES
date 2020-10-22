#! /bin/bash

rm results*

MODEL_TF=(BERT Roberta camamBert distilBert flauBert funnel bart long_former reformer xlm_roberta)
MODEL_ALLEN_NLP=(elmo_Bidaf Bidaf transformer_qna NAQANet)

readarray -t QUESTION <questions.txt
echo -n MODEL $'\t'>results.tsv
for questiontype in "${QUESTION[@]}"
	do
    	echo -n $questiontype $'\t'>> results.tsv
    done
echo >>	results.tsv


source "../env/bert-transformer/bin/activate"
which python3
for modeltype in "${MODEL_TF[@]}"
do
	echo -n $modeltype $'\t'>>results.tsv
	for questiontype in "${QUESTION[@]}"
	do
		answer=$(python3 tester_tf.py "$questiontype" $modeltype)
		echo -n $answer $'\t'>>results.tsv
	done
	echo >>	results.tsv
done
deactivate


source "../env/elmo-allen_nlp/bin/activate"
which python3
for modeltype in "${MODEL_ALLEN_NLP[@]}"
do
	echo -n	$modeltype $'\t'>>results.tsv
	for questiontype in "${QUESTION[@]}"
	do
		answer=$(python3 tester_allen_nlp.py "$questiontype" $modeltype)
		echo -n	$answer $'\t'>>results.tsv
	done
	echo >>	results.tsv
done
deactivate