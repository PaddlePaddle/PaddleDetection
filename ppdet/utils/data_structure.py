class BufferDict(dict):
    def __init__(self, **kwargs):
        super(BufferDict, self).__init__(**kwargs)

    def __getitem__(self, key):
        return super(BufferDict, self).__getitem__(key)

    def __setitem__(self, key, value):
        super(BufferDict, self).__setitem__(key, value)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def get(self, key):
        if key in self.keys():
            return self.__getitem__(key)
        else:
            raise "The %s is not in global inputs dict" % key

    def set(self, key, value):
        if key not in self.keys():
            self.__setitem__(key, value)
        else:
            raise "The %s is already in global inputs dict" % key
