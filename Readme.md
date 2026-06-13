# DiffLogic-MLP: Multi-Layer Perceptron using Differentiable Logic Gates

A project to build a neural network out of logic gates.

This repository contains a pure C implementation of a Multi-Layer Perceptron (MLP) trained using differentiable logic gates, featuring export via the AIGER format.

This work is directly inspired by the groundbreaking research of **Felix Petersen et al.** on hardware-efficient neural networks.

## Description

This project implements a **sparse MLP** in which each neuron has **fewer than 16 inputs**.

Each neuron is converted into a logic circuit using the **Quine–McCluskey algorithm**, producing an **AND network**.

Finally, all generated circuits are assembled together to build a large **AIGER** file.

The goal of this project is to train neural networks that can be exported and executed as digital logic circuits.

## Features

- Sparse MLP training on datasets using primitives defined in \`problem.h\`
- Example implementation using **MNIST**
- **AIGER export**
- **Interactive training**
- Sparse architecture with fewer than **16 inputs per neuron**
- Logic synthesis of neurons into AND networks using **Quine–McCluskey**
- Compatible with **ABC** for balancing and optimization

## Results

On **MNIST**, the project achieves:

- **87% accuracy**
- **54 layers**
- Around **84k AND gates**

## Usage

### Requirements

You need:

- `make`
- **AIGER software** installed
- **ABC** installed

### Compile

```bash
make
```

### Training

Training is performed using the primitives defined in `problem.h`.

An example configuration using **MNIST** is already provided.

### Export

After training, the network can be exported to an **AIGER** file. *(press G key in GUI)*

### Circuit Optimization

For better results, balance and optimize the generated circuit using **ABC**.

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


This project implements and expands upon the concepts introduced in the following landmark paper:

> **Felix Petersen, Christian Borgelt, Hilde Kuehne, and Oliver Deussen.** 
> *"Deep Differentiable Logic Gate Networks."* 
> Published at **NeurIPS 2022**. [arXiv:2210.08277](https://arxiv.org/abs/2210.08277)


## License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License**.

https://creativecommons.org/licenses/by-nc-sa/4.0/

## How it works

The pipeline is simple:

1. Train a sparse MLP
2. Convert each neuron into boolean logic using the **Quine–McCluskey algorithm**
3. Generate an **AND network**
4. Merge all neuron circuits together
5. Export the final network as an **AIGER** file
6. Optimize the circuit using **ABC**