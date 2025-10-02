import qiskit as qk
import sympy as sp
import cp33simu

# globales (perdón Niklaus)
num_qubits = 5  # número de qubits a simular
corridas = 30  # corridas del Aer de benchmark
qc1 = qk.QuantumCircuit(1, 0)
qc1.h(0)
qs1 = cp33simu.simu(qc=qc1)



def test_cupy_installed():
    global qs1
    assert qs1.cupy_installed()


def test_validate_cupy():
    global qs1
    assert qs1.validate_cupy()

