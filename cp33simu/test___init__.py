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
    # qs1 = cp33simu.simuMPI(qc=qc1)

    # assert qs1.cupy_enabled == False
    # assert qs1.qc_matrix_load() == True
    assert True  # no implementado

def test_gputopologia():
        result = False
        qs1 = cp33simu.simuGPUbajo(qc=qc1)
        assert qs1.cupy_enabled == True  # necesitamos GPU
        qs1.qc_matrix = None
        resultOK = qs1.qc_matrix_load()  # carga la matriz del circuito en la GPU
        assert resultOK
        qs1.instate_matrix = None
        resultOK = qs1.instate_matrix_load()
        assert resultOK
        qs1.GPUtopology=3.14
        assert qs1.GPUtopology is None
        qs1.GPUtopology = {"matrix":qs1.qc_matrix, "siga": "siga"}
        assert qs1.GPUtopology is not None
        resultOK = True
        assert resultOK
        assert resultOK


def xxxtest_simuGPUbajo_outstate_calculate():
    result = False
    qs1 = cp33simu.simuGPUbajo(qc=qc1)
    assert qs1.cupy_enabled == True  # necesitamos GPU
    qs1.qc_matrix = None
    resultOK = qs1.qc_matrix_load()  # carga la matriz del circuito en la GPU
    assert resultOK
    qs1.instate_matrix = None
    resultOK = qs1.instate_matrix_load()
    assert resultOK
    result = qs1.outstate_calculate()
    assert result


