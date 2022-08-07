class ComplexNumber(object):
    def __init__(self, re, im):
        self.re=re
        self.im=im
    def __add__(self, other):
        re=self.re+other.re
        im=self.im+other.im
        c=ComplexNumber(re,im)
        return c

    def __sub__(self, other):
        re=self.re-other.re
        im=self.im-other.im
        c=ComplexNumber(re,im)
        return c

    def __mul__(self, other):
        re=self.re*other.re-other.im*self.im
        im=self.re*other.im+other.re*self.im
        c=ComplexNumber(re,im)
        return c

    def __truediv__(self, other):
        a=self.re
        b=self.im
        c=other.re 
        d=other.im
        c=ComplexNumber(  (a*c+b*d)/(c**2 + d**2)  ,  (b*c-a*d)/(c**2 + d**2) )
        return c
    def __invert__(self):
        a=self.re
        b=self.im
        c=ComplexNumber(a/(a**2+b**2)  , -b/(a**2+b**2) )
        return c

    def __abs__(self):
        return ((self.re)**2+(self.im)**2)**(1/2)

    def __eq__(self, other):
        if self.re==other.re and self.im==other.im:
            return True
        else:
            return False

    def __repr__(self):
        if self.im<0:
            return str("{0:.2f}".format(self.re)) + " - " + str("{0:.2f}".format(-self.im)) + "i"
        else:
            return str("{0:.2f}".format(self.re)) + " + " + str("{0:.2f}".format(self.im)) + "i"

