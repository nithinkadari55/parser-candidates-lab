class Rect:
    def __init__(self,*a):
        if len(a)==0: v=[0.0,0.0,0.0,0.0]
        elif len(a)==1: v=[float(x) for x in list(a[0])]
        else: v=[float(x) for x in a]
        self.x0,self.y0,self.x1,self.y1=v
    def _empty(self): return self.x0>=self.x1 or self.y0>=self.y1
    def __iter__(self): return iter([self.x0,self.y0,self.x1,self.y1])
    def __getitem__(self,i): return [self.x0,self.y0,self.x1,self.y1][i]
    def __len__(self): return 4
    def get_area(self):
        return max(0.0,self.x1-self.x0)*max(0.0,self.y1-self.y0)
    def intersect(self,r):
        r=r if isinstance(r,Rect) else Rect(r)
        if self._empty() or r._empty():
            self.x0=self.y0=self.x1=self.y1=0.0; return self
        x0,y0,x1,y1=max(self.x0,r.x0),max(self.y0,r.y0),min(self.x1,r.x1),min(self.y1,r.y1)
        if x0>=x1 or y0>=y1: x0=y0=x1=y1=0.0
        self.x0,self.y0,self.x1,self.y1=x0,y0,x1,y1; return self
    def include_rect(self,r):
        r=r if isinstance(r,Rect) else Rect(r)
        if r._empty(): return self
        if self._empty():
            self.x0,self.y0,self.x1,self.y1=r.x0,r.y0,r.x1,r.y1; return self
        self.x0,self.y0=min(self.x0,r.x0),min(self.y0,r.y0)
        self.x1,self.y1=max(self.x1,r.x1),max(self.y1,r.y1); return self
