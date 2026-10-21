# DiffLogic-MLP: Multi-Layer Perceptron using Differentiable Logic Gates

A project to build a neural network out of logic gates.

This repository contains a pure C implementation of a Multi-Layer Perceptron (MLP) trained using differentiable logic gates, featuring export via the AIGER format.

This work is directly inspired by the groundbreaking research of **Felix Petersen et al.** on hardware-efficient neural networks.

## Description

This project implements a **sparse MLP** in which each neuron has **fewer than 16 inputs** and uses a tempered hyperbolic tangent as an activation function.

Each neuron is converted into a logic circuit using the **Quine–McCluskey algorithm** on the sign response table for all combinations of 1 and -1 as inputs, producing an **AND network**.

Finally, all generated circuits are assembled together to build a large **AIGER** file.

The goal of this project is to train neural networks that can be exported and executed as digital logic circuits.

## Structure

- `mlp.c`/`mlp.h`: Contains the implementation of the MLP (this is the core file).
- `qm16.c`/`qm16.h`: Contains an implementation of the Quine–McCluskey algorithm for tables with fewer than 16 inputs.
- `aiger.c`/`aiger.h`: A dependency to help write binary AIGER files.
- `main.c`: Contains an example of usage on the MNIST dataset.

## TO DO

- Implement Adam optimizer (currently changing `lr` manually during training).
- Test it on other datasets.
- Test it on a larger scale.
- Try to make it output a probability distribution.
  
## Results

On **MNIST**, the project achieved after a bit of training:

- **87% accuracy**
- **54 layers**
- Around **84k AND gates**

*Note: the tested file is `mnist_87.aig`*

## Usage

### Compile

```bash
make -DAIGSIM_PATH="aigsim"
```

`aigsim` can be found here: [https://github.com/arminbiere/aiger](https://github.com/arminbiere/aiger)

### Training

Training is performed using the primitives defined in `problem.h`.
An example configuration using **MNIST** is already provided.

### Export

After training, the network can be exported to an **AIGER** file. *(Press the `G` key in the GUI)*

### Circuit Optimization

For better results, balance and optimize the generated circuit using **ABC** ([https://github.com/berkeley-abc/abc](https://github.com/berkeley-abc/abc)).

Example:

```bash
abc
read out.aig
balance
rewrite
balance
write optimized.aig
```

## Reference

This project implements and expands upon the concepts introduced in the following paper:

> **Felix Petersen, Christian Borgelt, Hilde Kuehne, and Oliver Deussen.** 
> *"Deep Differentiable Logic Gate Networks."* 
> Published at **NeurIPS 2022**. [arXiv:2210.08277](https://arXiv.org/abs/2210.08277)

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License**.

[https://creativecommons.org/licenses/by-nc-sa/4.0/](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## How it works

The pipeline is simple:

1. Train a sparse MLP.
2. Convert each neuron into boolean logic using the **Quine–McCluskey algorithm**.
3. Generate an **AND network**.
4. Merge all neuron circuits together.
5. Export the final network as an **AIGER** file.
6. Optimize the circuit using **ABC**.
7. Use the network to do fun things.

ELLIS UnConference