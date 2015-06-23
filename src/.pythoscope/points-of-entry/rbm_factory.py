"""
point of entry for pythoscope auto test generation
"""
from blocks.factories import rbm_factory, einet_factory
rbm = rbm_factory([5, 10, 50, 10, 5])
ein = einet_factory([30, 100, 20])
