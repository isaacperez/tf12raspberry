#ifndef _IMAGEN_H_
#define _IMAGEN_H_

typedef float type_for_pixels;

type_for_pixels* crearImagen(type_for_pixels *img_ptr, const unsigned int filas, const unsigned int columnas, const unsigned int canales, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA);
void visualizarImagen(const type_for_pixels *img_ptr, const unsigned int filas, const unsigned int columnas, const unsigned int canales, const unsigned int STEP_FILA, const unsigned int STEP_COLUMNA);

#endif
