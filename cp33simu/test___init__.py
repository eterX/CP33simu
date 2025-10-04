import qiskit as qk
import sympy as sp
import cp33simu

# globales (perdón Niklaus)
num_qubits = 5  # número de qubits a simular
corridas = 30  # corridas del Aer de benchmark
qc1 = qk.QuantumCircuit(1, 0)
qc1.h(0)


def test_validate_cupy():
    global qc1
    qs1 = cp33simu.simuGPU(qc=qc1)
    assert qs1.cupy_enabled == qs1.validate_cupy()


def test_qc_matrix_load():
    qs1 = cp33simu.simuMPI(qc=qc1)

    assert qs1.cupy_enabled == False
