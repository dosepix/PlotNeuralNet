
import sys
sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    # to_input('MC_in_example.pdf', offset="(0,0,0)", to='(0,0,0)', width=8, height=8, name="input"),
    to_ConvRelu("conv1", (59, 3), 32, offset="(0,0,0)", to="(0,0,0)", height=3, depth=59, width=16),
    to_ConvRelu("conv2", (59, 3), 16, offset="(1.5,0,0)", to="(conv1-east)", height=3, depth=59, width=8),
    to_connection("conv1", "conv2"), 
    to_ConvRelu("conv3", (59, 3), 8, offset="(3,0,0)", to="(conv2-east)", height=3, depth=59, width=4),
    to_connection("conv2", "conv3"), 
    to_ConvRelu("conv4", (59, 3), 16, offset="(3,0,0)", to="(conv3-east)", height=3, depth=59, width=8),
    to_connection("conv3", "conv4"), 
    to_ConvRelu("conv5", (59, 3), 32, offset="(1.5,0,0)", to="(conv4-east)", height=3, depth=59, width=16),
    to_connection("conv4", "conv5"), 
    to_FcRelu("fc1", 512, offset="(3,0,0)", to="(conv5-east)", height=1, depth=120, width=2),
    to_connectionDashed(of="conv5", to="fc1"),
    to_FcRelu("fc2", 256, offset="(1,0,0)", to="(fc1-east)", height=1, depth=100, width=2),
    to_connectionDashed(of="fc1", to="fc2"),
    to_FcRelu("fc3", 120, offset="(1,0,0)", to="(fc2-east)", height=1, depth=80, width=2),
    to_connectionDashed(of="fc2", to="fc3"),
    # to_input('MC_out_example.pdf', offset="(3,0,0)", to='(fc3-east)', width=8, height=8, name="output"),
    to_end()
    ]
'''
to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)"),
to_Conv("conv2", 128, 64, offset="(1,0,0)", to="(pool1-east)", height=32, depth=32, width=2 ),
to_connection( "pool1", "conv2"), 
to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=28, depth=28, width=1),
to_SoftMax("soft1", 10 ,"(3,0,0)", "(pool1-east)", caption="SOFT"  ),
to_connection("pool2", "soft1"),    
'''

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
