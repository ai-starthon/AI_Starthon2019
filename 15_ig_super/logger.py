import nsml

class Logger:
    def __init__(self):        
        self.last = None
    
    def scalar_summary(self, tag, value, step):
        if self.last and self.last['step'] != step:
            nsml.report(**self.last, scope=None)
            self.last = None
        if self.last is None:
            self.last = {'step':step, 'iter':step, 'epoch':1}
        self.last[tag] = value

   