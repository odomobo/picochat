# plan is:
# 25M parameters, broken down as
# 12.5M embedding parameters
# 12.5M transformer parameters
# (initially 38.5M parameters, 12.5M wte, 12.5M transformers, 12.5M lm_head, but we'll tie them together once we get the basic functionality working)
#
# The plan for model architecture:
#
# vocab size   : 24576 (3 * 2^13)
# depth        : 4
# dim          : 512
# head dim     : 64
# heads        : 8
#
# calculations:
# vocab size * dim = 12.5M
# [ dim*dim*4 + [dim*(dim*4)]*2 ] * depth = 12.5M