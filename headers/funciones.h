#ifndef _FUNCIONES_H_
#define _FUNCIONES_H_
#include "imagen.h"

type_for_pixels* ReLU(type_for_pixels *img,  const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA, const unsigned int PADDING_SIZE);

type_for_pixels* Max_Pooling_2x2_S2(type_for_pixels* img, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA, const unsigned int STEP_FILA_OUTPUT, const unsigned int PADDING_SIZE_ENTRADA, const unsigned int PADDING_SIZE_SALIDA, type_for_pixels* output);
type_for_pixels* Max_Pooling_3x3_S2(type_for_pixels* img, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA, const unsigned int STEP_FILA_OUTPUT, const unsigned int PADDING_SIZE_ENTRADA, const unsigned int PADDING_SIZE_SALIDA, type_for_pixels* output);

type_for_pixels* suma_elemento_elemento(type_for_pixels* img1, const type_for_pixels* img2, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA);
type_for_pixels* suma_vector_con_tensor_por_canales(type_for_pixels* img, const type_for_pixels* vector, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA, const unsigned int tiene_padding_la_entrada);

type_for_pixels* batch_norm(type_for_pixels* img, const type_for_pixels* param_batch_norm, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA);

type_for_pixels* producto_tensores(const type_for_pixels* img, const type_for_pixels* weights, type_for_pixels* producto_tensores, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA, const unsigned int NUM_NEURONAS);

type_for_pixels* Avg_Pooling_2x2_S2(type_for_pixels* img, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA, const unsigned int STEP_FILA_OUTPUT, const unsigned int PADDING_SIZE_ENTRADA, const unsigned int PADDING_SIZE_SALIDA, type_for_pixels* output);
type_for_pixels* Avg_Pooling_Completo_por_canal(type_for_pixels* img, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA, const unsigned int PADDING_SIZE_ENTRADA, type_for_pixels* output);

type_for_pixels* Convolucion_3x3_S1(type_for_pixels* img, type_for_pixels *img_output, type_for_pixels *kernel, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA, const unsigned int STEP_FILA_OUTPUT, const unsigned int STEP_COLUMNA_OUTPUT, const unsigned int NUM_FILTROS);
type_for_pixels* Convolucion_1x1_S1(type_for_pixels* img, type_for_pixels *img_output, type_for_pixels *kernel, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA, const unsigned int STEP_FILA_OUTPUT, const unsigned int STEP_COLUMNA_OUTPUT, const unsigned int NUM_FILTROS);
type_for_pixels* Convolucion_3x3_S2(type_for_pixels* img, type_for_pixels *img_output, type_for_pixels *kernel, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA, const unsigned int STEP_FILA_OUTPUT, const unsigned int STEP_COLUMNA_OUTPUT, const unsigned int NUM_FILTROS);
type_for_pixels* Convolucion_1x1_S2(type_for_pixels* img, type_for_pixels *img_output, type_for_pixels *kernel, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA, const unsigned int STEP_FILA_OUTPUT, const unsigned int STEP_COLUMNA_OUTPUT, const unsigned int NUM_FILTROS);
type_for_pixels* Convolucion_7x7_S2(type_for_pixels* img, type_for_pixels *img_output, type_for_pixels *kernel, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA, const unsigned int STEP_FILA_OUTPUT, const unsigned int NUM_FILTROS);

type_for_pixels* add_padding_same_con_copia_imagen(type_for_pixels* img, type_for_pixels* img_padding, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA, const unsigned int PADDING_SIZE);
type_for_pixels* add_padding_same(type_for_pixels* img, const unsigned int FILAS, const unsigned int COLUMNAS, const unsigned int CANALES, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA, const unsigned int PADDING_SIZE);

#endif
