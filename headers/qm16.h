#ifndef QM16_H
#define QM16_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QM16_VARS 8
#define QM16_SPACE (1u << QM16_VARS)

typedef struct {
  uint32_t* data;
  size_t size;
  size_t capacity;
} qm16_vec_t;

/* =========================
   bit encoding of a terms
   =========================
   [31..16] value
   [15.. 0] mask
*/

/* =========================
   API principale
   ========================= */
int qm16_generate(qm16_vec_t* out, int (*gen)(void*, int), void* user_data);

/* free result */
void qm16_free(qm16_vec_t* v);


#ifdef __cplusplus
}
#endif

#endif