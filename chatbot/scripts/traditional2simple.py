# coding:utf8
# @Time    : 18-5-14 下午2:13
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import opencc

from chatbot.utils.log import get_logger
from chatbot.utils.wrapper import time_counter


logger = get_logger("Text Preprocessing T2S")


@time_counter
def traditional2simple(input_path, output_path):
    """繁体转简体

    :param input_path:
    :param output_path:
    :return:
    """
    # input check
    assert os.path.exists(input_path)
    # output check
    path = Path(output_path).resolve()
    path.parent.mkdir(exist_ok=True)
    # convert
    f_in = open(input_path, 'r', encoding='utf8')
    f_out = open(output_path, 'w', encoding='utf8')
    cc = opencc.OpenCC('t2s')
    for (i, line) in enumerate(f_in.readlines()):
        f_out.write(cc.convert(line))
        if i % 10000 == 0:
            logger.info("t2s %d lines complete" % i)
    f_in.close()
    f_out.close()
    logger.info("Finished Text T2S")


if __name__ == "__main__":
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument('-i', '--input', type=str)
    parse.add_argument('-o', '--output', type=str)
    args = parse.parse_args()
    print(args.input)
    traditional2simple(args.input, args.output)


