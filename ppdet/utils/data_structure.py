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

    def get(self, key):
        return self.__getitem__(key)

    def set(self, key, value):
        self.__setitem__(key, value)

    def debug(self, dshape=True, dtype=False, dvalue=False, name='all'):
        if name == 'all':
            ditems = self.items()
        else:
            ditems = self.get(name)

        for k, v in ditems:
            info = [k]
            if dshape == True and hasattr(v, 'shape'):
                info.append(v.shape)
            if dtype == True:
                info.append(type(v))
            if dvalue == True and hasattr(v, 'numpy'):
                info.append(np.mean(np.abs(v.numpy())))

            print(info)
