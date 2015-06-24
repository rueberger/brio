# brio
brio is a modular neural network library, inspired by [blocks](https://github.com/mila-udem/blocks).
It is designed to facilitate the construction and training of biologically plausible neural networks - brio provides
several local learning rules and makes it trivial to constraint weights and build networks that obey Dale's law

## Architecture
Neural networks with brio are constructed out of two atomic components: Layers and Connections. Network state is
divided intuitively between Layer and Connection objects; neuron state is stored in layer objects and connection
objects store the weights defining transformations between layers. There are essentially no restrictions on the ways
layers and connections can be composed. Recurrent networks can be constructed with ease.

## Disclaimer
brio is very much a work in progress. Don't hesitate to contact me with any bugs or questions
