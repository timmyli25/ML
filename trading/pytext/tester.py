import sys
import pytext

config_file = sys.argv[1]
model_file = sys.argv[2]
headline = sys.argv[3]

config = pytext.load_config(config_file)
predictor = pytext.create_predictor(config, model_file)

result = predictor({"raw_text": headline})
doc_label_scores_prefix = (
    'scores:' if any(r.startswith('scores:') for r in result)
    else 'doc_scores:'
)

print(max((label for label in result if label.startswith(doc_label_scores_prefix)),key=lambda label: result[label][0]))
gen = (label for label in result if label.startswith(doc_label_scores_prefix))
for i in gen:
    print(result[i][0])
best_doc_label = max(
        (label for label in result if label.startswith(doc_label_scores_prefix)),
        key=lambda label: result[label][0],
    # Strip the doc label prefix here
    )[len(doc_label_scores_prefix):]
