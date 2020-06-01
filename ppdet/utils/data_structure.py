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
        return self.__getitem__(key)

    def set(self, key, value):
        self.__setitem__(key, value)


'''
class BufferDict():
    """
    1. buffer key out to avoid pass argument;
    2. easy to debug, such as "print some tensor's shape" 
    """
    def __init__(self, ):
        self.dict = dict()

    def update(self, sub_dict):
        # TODO: add check conflict
        self.dict.update(sub_dict)

    def set(self, k, v):
        if k not in self.dict.keys():
            self.dict[k] = v 
        else:
            raise "The %s is already in global inputs dict"%k
         
    def get(self, name):
        if name in self.dict.keys():
            return self.dict[name]
        else:
            raise "The %s is not in global inputs dict"%name 

    def keys(self,):
        return self.dict.keys()
'''
