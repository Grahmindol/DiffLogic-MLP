#ifndef PROBLEM_H
#define PROBLEM_H

#define TRAIN_IMAGES_PATH "resources/mnist_train_images.bin"
#define TRAIN_LABELS_PATH "resources/mnist_train_labels.bin"
#define TEST_IMAGES_PATH "resources/mnist_test_images.bin"
#define TEST_LABELS_PATH "resources/mnist_test_labels.bin"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define INPUT_SIZE (27 * 27)
#define OUTPUT_SIZE 64

#define INIT_LR 0.0005
#define INIT_WTEMP 0.5
#define INIT_OTEMP 0.5
#define INIT_STEMP 0.01

#define INIT_STEPS_PER_FRAME 256

static const size_t layers[] = {INPUT_SIZE, 512, 512, 512, 512, 512, 512, OUTPUT_SIZE};
static const size_t nb_layers = sizeof(layers) / sizeof(layers[0]);

static uint8_t* train_images = NULL;
static uint8_t* train_labels = NULL;
static int train_count = 0;

static uint8_t* test_images = NULL;
static uint8_t* test_labels = NULL;
static int test_count = 0;

static uint32_t read_be32(FILE* f) {
  uint8_t b[4];

  if (fread(b, 1, 4, f) != 4) {
    fprintf(stderr, "Failed to read u32\n");
    exit(1);
  }

  return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | ((uint32_t)b[3]);
}

static void load_mnist(const char* image_path, const char* label_path, uint8_t** images_out,
                       uint8_t** labels_out, int* count_out) {
  FILE* fi = fopen(image_path, "rb");
  FILE* fl = fopen(label_path, "rb");

  if (!fi || !fl) {
    fprintf(stderr, "Failed to open MNIST files\n");
    exit(1);
  }

  uint32_t img_magic = read_be32(fi);
  uint32_t img_count = read_be32(fi);
  uint32_t rows = read_be32(fi);
  uint32_t cols = read_be32(fi);

  uint32_t lbl_magic = read_be32(fl);
  uint32_t lbl_count = read_be32(fl);

  (void)img_magic;
  (void)lbl_magic;

  if (img_count != lbl_count) {
    fprintf(stderr, "MNIST count mismatch\n");
    exit(1);
  }

  if (rows != 28 || cols != 28) {
    fprintf(stderr, "Unexpected MNIST image size\n");
    exit(1);
  }

  *count_out = (int)img_count;

  *images_out = malloc((size_t)(*count_out) * 28 * 28);
  *labels_out = malloc((size_t)(*count_out));

  fread(*images_out, 1, (size_t)(*count_out) * 28 * 28, fi);

  fread(*labels_out, 1, (size_t)(*count_out), fl);

  fclose(fi);
  fclose(fl);
}

void dataset_init(void) {
  load_mnist(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, &train_images, &train_labels, &train_count);

  load_mnist(TEST_IMAGES_PATH, TEST_LABELS_PATH, &test_images, &test_labels, &test_count);

  printf("Train: %d samples\n", train_count);
  printf("Test : %d samples\n", test_count);
}

void dataset_destroy(void) {
  if (train_images) free(train_images);
  if (train_labels) free(train_labels);
  if (test_images) free(test_images);
  if (test_labels) free(test_labels);

  train_images = NULL;
  train_labels = NULL;
  train_count = 0;

  test_images = NULL;
  test_labels = NULL;
  test_count = 0;
}



typedef struct {
  uint8_t expected;
} test_validator_t;

static void make_sample(uint8_t* images, uint8_t* labels, int count, bool* inputs,
                        uint8_t* expected_label) {
  int idx = rand() % count;

  const uint8_t* img = &images[idx * 28 * 28];

  *expected_label = labels[idx];

  for (int i = 0; i < INPUT_SIZE; ++i) {
    inputs[i] = img[i] > 127;
  }
}
void dataset_make_train_sample(bool* inputs, bool* outputs) {
  uint8_t label;

  make_sample(train_images, train_labels, train_count, inputs, &label);

  for (int i = 0; i < OUTPUT_SIZE; i += 8) {
    for (int j = 0; j < 4; j++) {
      outputs[i + j] = (label >> j) & 1;
    }

    for (int j = 0; j < 4; j++) {
      outputs[i + 4 + j] = !outputs[i + j];
    }
  }
}

void dataset_make_test_sample(bool* inputs, test_validator_t* validator) {
  make_sample(test_images, test_labels, test_count, inputs, &validator->expected);
}

bool dataset_validate_output(const test_validator_t* validator, const bool* outputs) {
  int votes[10] = {0};

  for (int i = 0; i < OUTPUT_SIZE; i += 8) {
    int v = 0;

    for (int j = 0; j < 4; j++) {
      v |= outputs[i + j] << j;
    }

    bool valid = true;

    for (int j = 0; j < 4; j++) {
      if (outputs[i + 4 + j] == outputs[i + j]) {
        valid = false;
        break;
      }
    }

    if (valid && v < 10) {
      votes[v]++;
    }
  }

  int best = 0;

  for (int i = 1; i < 10; i++) {
    if (votes[i] > votes[best]) best = i;
  }

  return best == validator->expected;
}

#endif