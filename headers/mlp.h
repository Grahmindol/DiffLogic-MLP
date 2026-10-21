#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>

#include "aiger.h"

#define NB_ENTRY 8

typedef struct {
  size_t nb_layers;
  size_t* sizes;

  double** z;
  double** a;
  double** delta;
  double** b;

  double*** W;  // poids [l][j][k]
  int*** I;     // index d'entrée [l][j][k]

  void* meta_block;
  double* data_block;
  int* idx_block;

  double weights_temp;
  double out_temp;
} MLP;

/**
 * @brief Allocates and initializes a new Multi-Layer Perceptron (MLP) structure.
 *
 * Uses contiguous memory blocks (`meta_block`, `data_block`, and `idx_block`) to hold
 * metadata pointers, activations, weights, biases, and index connections.
 *
 * @param nb_layers Total number of layers in the network (including input and output layers).
 * @param sizes Array specifying the number of neurons in each layer.
 * @param weights_temp Temperature scaling factor for weight logits.
 * @param out_temp Temperature scaling factor for output activations.
 * @return Fully allocated MLP structure, or a zero-initialized struct on failure.
 */
MLP mlp_create(size_t nb_layers, const size_t* sizes, double weights_temp, double out_temp);

/**
 * @brief Frees all dynamic memory blocks allocated for an MLP network and resets its fields to
 * zero.
 *
 * @param net Pointer to the MLP structure to be freed.
 */
void mlp_free(MLP* net);

/**
 * @brief Re-initializes all weights, biases, and connections of an existing MLP with random values.
 *
 * @param net Pointer to the MLP structure to randomize.
 */
void mlp_randomize(MLP* net);

/**
 * @brief Sets and clamps temperature parameters for weight transformation and output activation.
 *
 * Clamps temperatures between 0.05 and 100.0 to ensure numerical stability.
 *
 * @param net Pointer to the MLP structure.
 * @param weights_temp Temperature parameter for weight logit transformation.
 * @param out_temp Temperature parameter for output activation functions.
 */
void mlp_set_temps(MLP* net, double weights_temp, double out_temp);

/**
 * @brief Performs a forward pass through the network given an input array.
 *
 * Computes layer-by-layer pre-activation sums (z) and post-activation values (a)
 * using OpenMP multi-threading for intermediate layers.
 *
 * @param net Pointer to the MLP structure.
 * @param input Pointer to an array of size equal to the input layer size (`net->sizes[0]`).
 */
void mlp_forward(MLP* net, const double* input);

/**
 * @brief Performs backpropagation and updates weights and biases using stochastic gradient descent.
 *
 * Computes output error deltas, backpropagates error deltas through hidden layers, calculates
 * gradients, and updates weights/biases using the specified learning rate `lr`.
 *
 * @param net Pointer to the MLP structure.
 * @param target Pointer to array of target output values (size = `net->sizes[last_layer]`).
 * @param lr Learning rate for gradient descent weight/bias updates.
 * @return Average Mean Squared Error (MSE) loss for the current forward/backward step.
 */
double mlp_backward(MLP* net, const double* target, double lr);

/**
 * @brief Serializes and saves an MLP instance to a binary file.
 *
 * Binary file includes magic bytes ("MLP1"), versioning, network structure sizes, temperatures,
 * floating-point data parameters (weights, biases, activations), and connection indices.
 *
 * @param net Pointer to the MLP structure to save.
 * @param path File system path where the binary file will be saved.
 * @return 1 on success, or 0 on failure.
 */
int mlp_save(const MLP* net, const char* path);

/**
 * @brief Deserializes and loads an MLP instance from a binary file.
 *
 * @param path Path to the binary file containing a saved MLP structure.
 * @return Loaded MLP structure, or a zero-initialized struct if loading or header validation fails.
 */
MLP mlp_load(const char* path);

/**
 * @brief Exports the MLP neural network as an AIGER boolean logic circuit.
 *
 * Converts each neuron's behavior into a logic function using 16-variable Quine-McCluskey (qm16)
 * tabular reduction, constructs equivalent AND-Inverter Graphs (AIGs), and writes the circuit
 * using the AIGER library.
 *
 * @param net Pointer to the MLP structure.
 * @param mode AIGER file output format/mode (e.g., ascii or binary).
 * @param file Pointer to an open FILE handle for writing output.
 */
void mlp_aiger_export(const MLP* net, enum aiger_mode mode, FILE* file);

#endif