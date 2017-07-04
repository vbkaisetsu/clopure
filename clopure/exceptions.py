class ClopureSyntaxError(Exception):
    def __init__(self, *args, pos=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos = pos


class ClopureRuntimeError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
