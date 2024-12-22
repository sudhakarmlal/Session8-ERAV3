import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv_block1 = self.create_conv_block(3, 24, padding=1, dropout=0.01) # depthwise separable convolution layer
        self.trans_block1 = self.create_trans_block(24, 32, padding=0, dilation=1, dropout=0.01)
        self.conv_block2 = self.create_conv_block(32, 32, padding=1, dropout=0.01) # depthwise separable convolution layer
        self.trans_block2 = self.create_trans_block(32, 64, padding=0, dilation=2, dropout=0.01) # dilated kernels
        self.conv_block3 = self.create_conv_block(64, 64, padding=1, dropout=0.01) # depthwise separable convolution layer
        self.trans_block3 = self.create_trans_block(64, 96, padding=0, dilation=4, dropout=0.01) # dilated kernels
        self.conv_block4 = self.create_conv_block(96, 96, padding=1, dropout=0.01) # depthwise separable convolution layer
        self.trans_block4 = self.create_trans_block(96, 96, padding=0, dilation=8, dropout=0.01) # dilated kernels

        self.out_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(96, 10, 1, bias=True),
            nn.Flatten(),
            nn.LogSoftmax(-1)
        )


    @staticmethod
    def create_conv_block(input_c, output_c, padding=1, dropout=0, bias=False):

        if input_c == output_c:
            return nn.Sequential(
                nn.Conv2d(input_c, output_c, 3, groups=output_c, padding=1, bias=False),
                nn.BatchNorm2d(output_c),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Conv2d(output_c, output_c, 1, groups=1, padding=0, bias=False),

                nn.Conv2d(output_c, output_c, 3, groups=output_c, padding=1, bias=False),
                nn.BatchNorm2d(output_c),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Conv2d(output_c, output_c, 1, groups=1, padding=0, bias=False)
            )
        
        else:
            return nn.Sequential(
                nn.Conv2d(input_c, output_c, 3, groups=1, padding=1, bias=False),
                nn.BatchNorm2d(output_c),
                nn.Dropout(dropout),
                nn.ReLU(),

                nn.Conv2d(output_c, output_c, 3, groups=output_c, padding=1, bias=False),
                nn.BatchNorm2d(output_c),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Conv2d(output_c, output_c, 1, groups=1, padding=0, bias=False)
            )


    @staticmethod
    def create_trans_block(input_c, output_c, bias=False, padding=0, dilation=1, dropout=0):
        return nn.Sequential(
            nn.Conv2d(input_c, output_c, 3, bias=bias, padding=padding, dilation=dilation),
            nn.BatchNorm2d(output_c),
            nn.Dropout(dropout),
            nn.ReLU()
        )
    
    
    def forward(self, x):

        x = self.conv_block1(x)
        x = self.trans_block1(x)
        x = x + self.conv_block2(x)
        x = self.trans_block2(x)
        x = x + self.conv_block3(x)
        x = self.trans_block3(x)
        x = x + self.conv_block4(x)
        x = self.trans_block4(x)
        x = self.out_block(x)

        return x