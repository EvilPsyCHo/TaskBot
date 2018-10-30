# coding:utf8
# @Time    : 18-5-15 下午3:08
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from chatbot.preprocessing.word2vec import Word2vecExt


def main(mode, file_path=None, model_load_path=None, model_save_path=None):
    w2v = Word2vecExt()
    if mode == "build":
        w2v.build(file_path)
        w2v.save(model_save_path)
    elif mode == "update":
        w2v.load(model_load_path)
        w2v.update(file_path)
        w2v.save(model_save_path)
    else:
        print("一切都是幻觉")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", default=None)
    parser.add_argument("-l", "--load_path", default=None)
    parser.add_argument("-s", "--save_path", default=None)
    parser.add_argument("-t", "--text_path", default=None)
    args = parser.parse_args()
    main(args.mode, args.text_path, args.load_path, args.save_path)


