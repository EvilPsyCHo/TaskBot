# -*- coding: utf-8 -*-
# @Time    : 5/11/18 14:12
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com

from pathlib import Path


ROOT_PATH = Path(Path(__file__).resolve().parent, "..").resolve()
MODEL_PATH = ROOT_PATH / "results"
LOG_PATH = ROOT_PATH / "log"
