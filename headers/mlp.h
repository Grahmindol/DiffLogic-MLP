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

MLP mlp_create(size_t nb_layers, const size_t* sizes, double weights_temp, double out_temp);
void mlp_free(MLP* net);
void mlp_randomize(MLP* net);
void mlp_set_temps(MLP* net, double weights_temp, double out_temp);

void mlp_forward(MLP* net, const double* input);
double mlp_backward(MLP* net, const double* target, double lr);

int mlp_save(const MLP* net, const char* path);
MLP mlp_load(const char* path);
void mlp_aiger_export(const MLP* net, enum aiger_mode mode, FILE* file);

#endif