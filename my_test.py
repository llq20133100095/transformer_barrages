from hparams import Hparams
from nltk.translate.bleu_score import corpus_bleu
references = [[['▁怎么不开车', '去捡', '啊', '哦', '去']], [['▁怎么不开车', '去捡', '啊', '哦', '去']]]
candidates = [['▁怎么不开车', '去捡'], ['去捡', 'ni']]
score = corpus_bleu(references, candidates, weights=(0.5, 0.5, 0, 0))
print(score)

def read_file(file_name, sen_len):
    with open(file_name, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if line:
                sen_len.append(len(line.split()))
            else:
                break
    return sen_len

if __name__ == "__main__":
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    sens_len = []

    sens_len = read_file(hp.train_x, sens_len)
    sens_len = read_file(hp.eval_x, sens_len)
