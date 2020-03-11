import numpy

numpy.set_printoptions(threshold=numpy.nan)


def suma_columnas(A):
    # devuelve un vector con la suma de cada columna
    sumas = []
    for j in range(A.shape[1]):
        valor = 0
        for i in range(A.shape[0]):
            valor += A[i, j]
        sumas.append(valor)
    return sumas


def suma_filas(A):
    return suma_columnas(A.transpose())


def abs_matrix(A):
    # devuelve una matriz con los valores absolutos de A
    rows = A.shape[0]
    cols = A.shape[1]
    abs_A = numpy.zeros(shape=A.shape)
    for i in range(rows):
        for j in range(cols):
            abs_A[i, j] = abs(A[i, j])
    return abs_A


def busca_col_maxima(A):
    # devuelve el índice de la(s) columna(s) con el valor máximo
    sumas = suma_columnas(A)
    i = max(sumas)
    return numpy.where(sumas == i)[0]


def busca_fila_maxima(A):
    return busca_col_maxima(A.transpose())


def norma_1(A):
    return max(suma_columnas(abs_matrix(A)))


def norma_inf(A):
    return norma_1(A.transpose())


A = numpy.array([[1, 2, 3, -4, 4], [-4, 5, -6, -7, 7],
                 [-8, 9, -9, 5, 8], [7, 9, 10, 8, -5]])
abs_A = abs_matrix(A)
long_vectores_e = A.shape[1]


def signum(numero):
    if(numero < 0):
        return -1
    elif(numero > 0):
        return 1
    else:
        return 0


def vectores_norma_1_alcanzada(A):
    indices_norma_alcanzada = busca_col_maxima(abs_matrix(A))
    vectores_norma_alcanzada = []
    for i in indices_norma_alcanzada:
        vector_alc = list(numpy.zeros(shape=A.shape[1]))
        vector_alc[i] = 1
        vectores_norma_alcanzada.append(vector_alc)
    return vectores_norma_alcanzada


def vectores_norma_inf_alcanzada(A):
    indices_norma_alcanzada = busca_fila_maxima(abs_matrix(A))
    vectores_norma_alcanzada = []
    for i in indices_norma_alcanzada:
        vectores_norma_alcanzada.append(list(map(signum, A[i])))
    return vectores_norma_alcanzada


print(A)

print(suma_filas(abs_A))

print(busca_fila_maxima(abs_A))

print(vectores_norma_inf_alcanzada(A))


# print(norma_inf(A))
# print(numpy.linalg.norm(A, ord=numpy.inf))
