class bar(object):

    def __init__(self, prop):
        self.bar_prop = prop

    def get_prop(self):
        print 'bar prop is %s' % self.bar_prop


class foo(bar):

    def __init__(self, prop):
        super(foo, self).__init__('bar')
        self.foo_prop = 'foo'

    def get_prop(self):
        super(foo, self).get_prop()
        print 'foo prop is %s' % self.foo_prop


FOO = foo('bar')
FOO.get_prop()
