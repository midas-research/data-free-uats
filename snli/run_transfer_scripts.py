python snli-transfer.py -g "contradiction" -t "entailment" -r "1" -oh "0" -df "1"
python snli-transfer.py -g "contradiction" -t "neutral" -r "1" -oh "0" -df "1"

python snli-transfer.py -g "entailment" -t "contradiction" -r "1" -oh "0" -df "1"
python snli-transfer.py -g "entailment" -t "neutral" -r "1" -oh "0" -df "1"

python snli-transfer.py -g "neutral" -t "contradiction" -r "1" -oh "0" -df "1"
python snli-transfer.py -g "neutral" -t "entailment" -r "1" -oh "0" -df "1"
