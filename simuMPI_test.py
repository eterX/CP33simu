#
# simuMPI_test.py - Prueba de entorno MPI
#
# Este script es para validar que el stack MPI (mpi4py) está instalado
# y funcionando correctamente. Prueba comunicación básica entre procesos
# usando el simulador simuMPI.
#
# Uso (desde shell):
#   mpiexec -n 8 python -m mpi4py simuMPI_test.py
#
# donde -n 8 es el número de procesos MPI (ajustar según necesidad)
#
import cp33simu
import qiskit as qk
import mpi4py
from  mpi4py import MPI
# globales (perdón Niklaus)
num_qubits = 5  # número de qubits a simular
corridas = 30  # corridas del Aer de benchmark

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    qc1 = qk.QuantumCircuit(1, 0)
    qc1.h(0)
    simu = cp33simu.simuMPI(qc=qc1) # lo necesito acá para compartir dtype,etc

    # passing MPI datatypes explicitly
    if rank == 0:
        data =  cp33simu.cupy.arange(10, dtype=simu.cupy_dtype)
        comm.Send([data, simu.MPI_dtype], dest=1, tag=99)
    elif rank == 1:
        data = cp33simu.cupy.empty(10, dtype=simu.cupy_dtype)
        comm.Recv([data, simu.MPI_dtype], source=0, tag=99)
        print("Received:", data)
