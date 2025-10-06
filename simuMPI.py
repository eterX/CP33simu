#
# simuMPI.py - Similación de circuito cuantico en entorno MPI
#
# Uso (desde shell):
#   mpiexec -n XXXX python -m mpi4py simuMPI_test.py
#
# donde -n XXX es el número de procesos MPI (ajustar según necesidad)
#

import cp33simu
import qiskit as qk
import mpi4py
from  mpi4py import MPI

# globales (perdón Niklaus)
num_qubits = 5#3#5#2  # número de qubits a simular TODO: tomar como argumento
corridas = 30  # corridas del Aer de benchmark

def producto(vector1: cp33simu.cupy.array, vector2:cp33simu.cupy.array, fila):
    """
    producto interno entre dos vectores,

    :param vector1: vector fila
    :param vector2: vector columna
    :param fila: fila de la matriz del circuito q estamos calculando
    :return: producto interno entre los dos vectores <v_1|v_2>
    :rtype: cp33simu.cupy.array
    """
    try:
        #result = cp33simu.cupy.inner(vector1, vector2)
        result = cp33simu.cupy.dot(vector1, vector2) ##evita conjugación compleja de inner()
    except ValueError as e:
        print(f"ERROR: Error de dimensiones en producto interno: {e}, file {fila}")
        result = cp33simu.cupy.nan
    except Exception as e:
        print(f"ERROR: Error en producto interno {e}, file {fila}")
        result = cp33simu.cupy.nan
    return result

if __name__ == '__main__':

    if MPI.COMM_WORLD.Get_size() !=1 and MPI.COMM_WORLD.Get_size() != 2**num_qubits: #es 1 cunado debugueamos
        raise ValueError(f"""
los procesos deben ser de 2 elevado la cantidad de qubits. Para num_qubits={num_qubits} Correr:
mpiexec -n {2**num_qubits} python -m mpi4py simuMPI_test.py
                         """)
    qc1 = qk.QuantumCircuit(num_qubits, 0)
    qc1.h(range(num_qubits)) # preparo |+...+>
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    benchmarkDict =  {"qiskit-aer":False, "simuMPI":True, "simuGPU":True, "simuGPUbajo":True} # copio benchmarks_requeridos de main.ipynb
    simu = cp33simu.simuMPI(qc=qc1,benchmark=benchmarkDict) # lo necesito acá para compartir dtype,etc

    if rank == 0:
        # saca la matriz desde qiskit
        # para preparar |+...+> la mariz del circuito será (H|0⟩)⊗...⊗(H|0⟩)
        #
        simu.qc_matrix = None
        resultOK = simu.qc_matrix_load()  # carga la matriz del circuito en la GPU
        # resultOK=False
        if resultOK:
            pass  # TODO: mostramos matriz?
        else:
            msg = "no se pudo cargar la matriz del circuito"
            print(f"ERROR: {msg}")
            raise ValueError(msg)

        # preparamos el estado inicial
        simu.instate_matrix = None
        resultOK = simu.instate_matrix_load()
        if not resultOK:
            msg = "no se pudo cargar el estado de entrada"
            raise ValueError(f"ERROR: {msg}")

        # Preparamos buffers para Bcast (deben ser los arrays reales, no copias temporales)
        # Aplanamos el estado inicial de (2^n,1) a (2^n,) y guardamos en la variable
        simu.instate_matrix = simu.instate_matrix.flatten()
    else:
        # Creamos buffer receptor 1D para los workers
        simu.instate_matrix = cp33simu.cupy.empty(2**num_qubits, dtype=simu.cupy_dtype)

    # Broadcast del estado inicial (ahora simu.instate_matrix es el buffer real)
    # https://mpi4py.readthedocs.io/en/stable/reference/mpi4py.MPI.Comm.html#mpi4py.MPI.Comm.bcast
    comm.Bcast([simu.instate_matrix, 2**num_qubits, simu.MPI_dtype], root=0)

    # largamos el equivalente a simu.outstate_calculate()
    # es una matmul roñosa... perdón Alkarismi pero vamos a reutilizar el ejericio mpi_matrix
    if rank == 0:
        if simu.benchmark is not None:
            import time
            start = time.perf_counter()

        outstate = cp33simu.cupy.empty(2**num_qubits, dtype=simu.cupy_dtype)
        # Rank 0 calcula su propia fila (fila 0) localmente
        outstate[0] = producto(simu.qc_matrix[0], simu.instate_matrix, 0)
        for filaIdx in range(1, 2**num_qubits):
            # Distribuye filas 1+ a los workers (ranks 1 a 2^n-1)
            rank_dest = filaIdx  # Ahora rank i calcula fila i
            comm.Send([simu.qc_matrix[filaIdx], 2**num_qubits, simu.MPI_dtype],
                      dest=rank_dest, tag=1)

        for _ in range(1, 2**num_qubits): # Recibe resultados de los workers
            status = MPI.Status()
            valor = comm.recv(source=MPI.ANY_SOURCE, status=status, tag=3)
            outstate[status.Get_source()] = valor  # Rank i calculó fila i


    elif rank >0:
        # para los demas procesos "workers"
        # recibimos una fila del qc_matrix
        qc_matrix_fila = cp33simu.cupy.empty(2**num_qubits, dtype=simu.cupy_dtype)
        comm.Recv([qc_matrix_fila, 2**num_qubits, simu.MPI_dtype], source=0, tag=1)
        ## recibimos instate_matrix, ya no. ahora bcast'd
        #instate_matrix = cp33simu.cupy.empty(2**num_qubits, dtype=simu.cupy_dtype)
        #comm.Recv([instate_matrix, simu.MPI_dtype], source=0, tag=2)
        print(f"DEBUG: rank:{rank}, Received: {qc_matrix_fila}", )
        result = producto(qc_matrix_fila, simu.instate_matrix, rank)
        print(f"DEBUG: rank:{rank}, Resultado: {result}")
        comm.send(result, dest=0, tag=3)

    if rank == 0:
        if simu.benchmark is not None:
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000
            print(f"INFO: Benchmark completado en {elapsed_ms:.2f} ms")
            simu.benchmark.update({"walltime": elapsed_ms})
            print(f"DEBUG: simu.benchmark: {simu.benchmark}")
        # si llegamos hasta acá, nos ganamos un lugar con Gardel y Lepera :D
        # sacamos el estado por pantalla
        print(f"DEBUG:  simuPMI.utstate: {outstate}")
        print(f"\nINFO: Estado de salida - amplitud de probabilidad de los {2**num_qubits}   posibles estados::")
        for i, amp_proba in enumerate(outstate):#cp33simu.cupy.asnumpy(outstate)):
            estado_bin = format(i, f'0{num_qubits}b') # bin
            print(f"|{estado_bin}>: {amp_proba.real:.3f} +i{amp_proba.imag:.3f}")