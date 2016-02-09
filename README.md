# brio
brio is a modular neural network library, inspired by [blocks](https://github.com/mila-udem/blocks).
It is designed to facilitate the construction and training of biologically plausible neural networks - brio provides
several local learning rules and makes it trivial to constrain weights and build networks that obey Dale's law

## Architecture
Neural networks with brio are constructed out of two atomic components: Layers and Connections. Network state is
divided intuitively between Layer and Connection objects; neuron state is stored in layer objects and connection
objects store the weights defining transformations between layers. There are essentially no restrictions on the ways
layers and connections can be composed. Recurrent networks can be constructed with ease.

## Interactive tutorial

Learn how to use brio by following an example. iPython notebook [here](https://github.com/rueberger/brio/blob/master/examples/SAILnet.ipynb).

Can be viewed directly on github at the expense of the cool interactive bits. Best viewed in a local notebook session.

## Disclaimer

brio is, at heart, a research platform. Although I do my best to keep brio true to its original design, it occasionally suffers from the same feature creep and ad-hoc fixes that plague all academic code bases. It remains a work in progress.

If you've encountered a bug or have suggestions for how brio's architecture might be improved please don't hesitate in contacting me or filing an issue. 

