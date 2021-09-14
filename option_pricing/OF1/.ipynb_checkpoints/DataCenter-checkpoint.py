# -*- coding:utf-8 -*-
import datetime
import numpy as np
import math,time
import pandas as pd

import requests,sqlite3


class DataCenter():
    def __init__(self,db=False):
        self.session=requests.session()
        self.dbconn=SqliteConnection(db) if db else None
        pass
    def autoupdate(self):
        pass
    def __apiCommand(self,command):
        response=self.session.get(command)
        if response.status_code==200:
            return response.text
        else:
            raise ValueError('Requests Error Code:%d'%response.status_code)

    def get(self,via='API',**varname):
        #this function is to connect the df and programe
        """
        this function is to connect the df and programe
        """
        if  via=='sql':
            if self.dbconn is None:
                raise ValueError('No db given')
            elif 'Table' in varname.keys():
                tablename=varname.pop('Table')
                var=','.join(varname)
                return pd.read_sql('select %s in %s'%(var,tablename),self.dbconn)
            else:
                raise ValueError('Unknown sql info %s'%','.join(varname.values))
            pass
        elif via=='API':
            """
            构造api请求信息url
            """
            command='url'+','.join(varname.values())
            return self.__apiCommand(command)


def SqliteConnection(self,target):
        import sqlite3
        if target != 'default':
            conn = sqlite3.connect(target,timeout=10)
        else:
            conn = sqlite3.connect(":memory:")
        return conn
if __name__ == '__main__':
    pass
