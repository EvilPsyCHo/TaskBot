# coding:utf8
# @Time    : 18-5-15 下午1:54
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from chatbot.preprocessing.text import cut
from chatbot.utils.wrapper import  time_counter


@time_counter
def main(input_path, output_path, n_job):
    """文本文件分词

    要求文件以\n断句，不含标点及特殊字符
    分词后的文本以" "作为分割符号

    :param input_path:
    :param output_path:
    :return:
    """
    assert os.path.exists(input_path)
    out_path = Path(output_path).resolve().parent
    out_path.mkdir(exist_ok=True)
    in_file = open(input_path, 'r', encoding='utf8')
    out_file = open(output_path, 'w', encoding='utf8')
    text = in_file.readlines()
    text_cut = cut(text, n_job=n_job)
    for i in text_cut:
        out_file.write(" ".join(i[:-1]) + "\n")
    in_file.close()
    out_file.close()


if __name__ == "__main__":
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument("-i", "--input")
    parse.add_argument("-o", "--output")
    parse.add_argument("-n", "--n_job", default=4, type=int)
    args = parse.parse_args()
    main(args.input, args.output, args.n_job)

