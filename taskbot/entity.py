# -*- coding: utf-8 -*-
# @Time    : 7/7/18 14:01
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import datetime as dt

from taskbot.core.base import MetaEntity

__all__ = ["TimeInterval", "Location", "Organization"]


class TimeInterval(MetaEntity):

    def __init__(self):
        now = dt.datetime.now().date()
        super().__init__(
            start=str(dt.datetime(year=now.year, month=now.month, day=1).date()),
            end=str(now)
        )


class Location(MetaEntity):

    def __init__(self):
        super().__init__(
            province=None,
            city=None
        )


class Organization(MetaEntity):
    def __init__(self):
        super().__init__(
            alias=None,
            name=None,
            id=None,
        )
