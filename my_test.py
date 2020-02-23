from hparams import Hparams


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
