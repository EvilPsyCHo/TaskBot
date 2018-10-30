# -*- coding: utf-8 -*-
# @Time    : 5/12/18 13:22
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from gensim.corpora import WikiCorpus
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from chatbot.utils.log import get_logger
from chatbot.utils.wrapper import time_counter

logger = get_logger("Wiki Extract")


@time_counter
def wiki_extract(input_file, output_file):
    """wiki下载文件提取文本内容脚本

    :param input_file: 原始文件路径
    :param output_file:  提取文件路径
    :return: None
    """
    # 原始文件是否存在
    assert Path(input_file).resolve().exists()
    # 提取文件路径不存在就新建
    output_file_path = Path(output_file).resolve()
    output_file_path.parent.mkdir(exist_ok=True)
    logger.info("Start extract wiki ..")
    wiki = WikiCorpus(input_file, lemmatize=False)
    with open(output_file, "w", encoding="utf8") as f:
        for i, text in enumerate(wiki.get_texts()):
            f.write(" ".join(text) + "\n")
            if i % 10000 == 0:
                logger.info("Saved %d articles" % i)
    logger.info("Finished extract wiki, Saved in %s" % output_file)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("-o", "--output", type=str)
    args = parser.parse_args()
    wiki_extract(args.input_file, args.output_file)
