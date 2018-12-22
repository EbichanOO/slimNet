import chainer
import chainer.links as L
import chainer.functions as F
class SlimNet(chainer.Chain):
    def __init__(self, in_ch, out_ch, batchsize):
        super(SlimNet, self).__init__()
        self.out_ch = out_ch
        self.batch = batchsize

        with self.init_scope():
            self.conv1 = L.Convolution2D(in_ch, 8)
            self.value = Linear3D(in_ch, in_ch)
    
    def __call__(self, x):

            
class Linear3D(L.Linear):
    def __init__(self, *args, **kwargs):
        super(Linear3D, self).__init__(*args, **kwargs)
    def call(self, x):
        return super(Linear3D, self).__call__(x)
    def __call__(self, x):
        if x.ndim == 2:
            return self.call(x)
        assert x.ndim == 3

        x_2d = x.reshape((-1, x.shape[-1]))
        out_2d = self.call(x_2d)
        out_3d = out_2d.reshape(x.shape[:-1]+(out_2d.shape[-1], ))
        return out_3d