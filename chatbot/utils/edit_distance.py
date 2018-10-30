# coding:utf8
# @Time    : 18-7-4 上午9:19
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import numpy as np


def _substring_edit_distance(key_word, query):
    """计算子串最小编辑距离函数，包含删除/添加/替换三种编辑操作

    Args:
        key_word:
        query:

    Returns:
        int

    Example:

    >>> isinstance(_substring_edit_distance("来福士", "sdf来福斯蒂芬"), int)
    True

    >>> _substring_edit_distance("abc", "c aba c")
    1

    >>> _substring_edit_distance("来福士", "sdf福士斯蒂芬")
    1

    """
    len_query = len(query)
    len_keyword = len(key_word)
    matrix = np.zeros([len_keyword+1, len_query+1])
    matrix[:, 0] = range(len_keyword+1)
    for row in range(len_keyword):
        for col in range(len_query):
            replace = 0 if query[col] == key_word[row] else 1
            compare = min(
                matrix[row][col+1] + 1,
                matrix[row+1][col] + 1,
                matrix[row][col] + replace,
            )
            matrix[row+1][col+1] = compare
    return int(min(matrix[-1, :]))


class FuzzyMatch(object):
    def __init__(self, keywords, threshold):
        """

        Args:
            keywords:
            threshold: int or list
        """
        if isinstance(threshold, int):
            pass
        elif isinstance(threshold, list):
            assert len(threshold) == len(keywords)
            assert isinstance(threshold[0], int)
        else:
            raise ValueError
        self.threshold = threshold
        self.keywords = keywords

    def match(self, query):
        """

        Args:
            query:

        Returns:
            list of string

        """
        rst = []
        for idx, w in enumerate(self.keywords):
            ed = _substring_edit_distance(w, query)
            if isinstance(self.threshold, int):
                if ed <= self.threshold:
                    rst.append(w)
                else:
                    pass
            else:
                if ed <= self.threshold[idx]:
                    rst.append(w)
                else:
                    pass
        return rst


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())
    m = FuzzyMatch(["来福士", "仁恒"], [1, 0])
    m.match("哈哈来副士")
    m.match("仁横用电")
