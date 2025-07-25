# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 17:14:25 2025

@author: u6942852
"""

import numpy as np
from csv import writer
from os import remove, mkdir
from os.path import exists
from pathlib import Path
from shutil import copyfile
from collections.abc import Collection

class Fileprinter:
    def __init__(
            self, 
            file_name: str, 
            save_freq: int, 
            header: Collection[str] = None, 
            resume: bool = False,
            create_dir: bool = True
            ):
        self.file_name=file_name
        self.temp_file_path = '-temp.'.join(self.file_name.split('.'))
        self.save_freq=save_freq
        self.callno = 0
        self.array = None
        if resume is False:
            self._createfile(header, create_dir)
        if resume is True:
            try:
                copyfile(self.file_name, self.temp_file_path)
                remove(self.temp_file_path)
            except FileNotFoundError:
                self._createfile(header, create_dir)
                  
    def Terminate(self):
        if self.array is not None:
            self._flush()
                      
    def __call__(self, arr):
        self.callno+=1     
        if self.array is None:
            self.array = np.atleast_2d(arr)
        else: 
            self.array = np.concatenate((self.array, arr), axis=0)
        if self.callno % self.save_freq == 0:
            self._flush()
    
    def _print(self):
        with open(self.temp_file_path, 'a', newline='') as file:
            writer(file).writerows(self.array) 
            file.close()
    
    def _copyfile(self, forward=True):
        if forward is True:
            try:
                copyfile(self.file_name, self.temp_file_path)
            except FileNotFoundError as e:
                if self.callno == self.save_freq:
                    pass
                else: 
                    raise e 
        else:
           copyfile(self.temp_file_path, self.file_name)
           remove(self.temp_file_path)
           
    def _flush(self):
        print('\rWriting out to file. Do not interrupt', end='\r')
        self._copyfile(True)
        self._print()
        self._copyfile(False)
        print('\r'+' '*40, end='\r')
        self.array=None
    
    def _createfile(self, header, create_dir):
        p = Path(self.file_name)
        if create_dir is False: 
            assert exists(p.parent), "Parent folder does not exist. Pass `create_dir=True` to enable creation"
        else: 
            if not exists(p.parent):
                mkdir(p.parent)
        
        if header is not None:
            with open(self.file_name, 'w', newline='') as file:
               writer(file).writerow(header)
               file.close()
        else: 
            with open(self.file_name, 'w', newline='') as file:
               writer(file)
               file.close()

        