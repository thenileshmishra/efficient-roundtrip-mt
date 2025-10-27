"""Quick chrF++ test using SacreBLEU."""
from sacrebleu.metrics import CHRF

references = [
    "Ukham luräwinakax kuntix wakunanakax qullqaspan uk uñacht’ayasitayna ukatx juk’amp uka qullanakax ch’iqintarakitayna."
]
predictions = [
    "Ukham luräwinakax kuntix wakunanakax qullqaspan uk uñacht’ayasitayna ukatx juk’amp uka qullanakax ch’iqintarakitayna."
]

# chrF++ is chrF with word n-grams; set word_order=2 (char_order defaults to 6)
metric = CHRF(word_order=2, char_order=6)
score = metric.corpus_score(predictions, [references])

print(f"chrF++: {score.score:.3f}")
