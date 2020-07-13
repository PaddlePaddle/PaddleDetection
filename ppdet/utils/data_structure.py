import numpy as np


class BufferDict(dict):
    def __init__(self, **kwargs):
        super(BufferDict, self).__init__(**kwargs)

    def __getitem__(self, key):
        if key in self.keys():
            return super(BufferDict, self).__getitem__(key)
        else:
            raise Exception("The %s is not in global inputs dict" % key)

    def __setitem__(self, key, value):
        if key not in self.keys():
            super(BufferDict, self).__setitem__(key, value)
        else:
            raise Exception("The %s is already in global inputs dict" % key)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def update_v(self, key, value):
        if key in self.keys():
            super(BufferDict, self).__setitem__(key, value)
        else:
            raise Exception("The %s is not in global inputs dict" % key)

    def get(self, key):
        return self.__getitem__(key)

    def set(self, key, value):
        return self.__setitem__(key, value)

    def debug(self, dshape=True, dvalue=True, dtype=False):
        if self['open_debug']:
            if 'debug_names' not in self.keys():
                ditems = self.keys()
            else:
                ditems = self['debug_names']

            infos = {}
            for k in ditems:
                if type(k) is dict:
                    i_d = {}
                    for i, j in k.items():
                        if type(j) is list:
                            for jj in j:
                                i_d[jj] = self.get_debug_info(self[i][jj])
                        infos[i] = i_d
                else:
                    infos[k] = self.get_debug_info(self[k])
            print(infos)

    def get_debug_info(self, v, dshape=True, dvalue=True, dtype=False):
        info = []
        if dshape == True and hasattr(v, 'shape'):
            info.append(v.shape)
        if dvalue == True and hasattr(v, 'numpy'):
            info.append(np.mean(np.abs(v.numpy())))
        if dtype == True:
            info.append(type(v))
        return info
