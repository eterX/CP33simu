import qiskit as qk
import sympy as sp
import cp33simu
import pandas as pd
display = print

# globales (perdón Niklaus)
num_qubits = 5  # número de qubits a simular
corridas = 30  # corridas del Aer de benchmark
qc1 = qk.QuantumCircuit(1, 0)
qc1.h(0)
benchmarks_columns = ["clase",  # "qiskit-aer" o implementacion de simuabstracto
                      "circuito",
                      "corrida",
                      "num_qubits",
                      "walltime"]

benchmarks_corridos = pd.DataFrame(columns=benchmarks_columns)

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


def test_simuGPUbajo_outstate_calculate():
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



def test_simuGPU_benchmarks():
        global benchmarks_corridos
        result = False
        corridasGPU=2
        benchmarks_requeridos = {"qiskit-aer": False, "simuMPI": True, "simuGPU": True,
                                 "simuGPUbajo": True}  # Todo validar claves, o no :)
        benchmark_actual=benchmarks_requeridos.copy()
        qs1 = cp33simu.simuGPUbajo(qc=qc1,
                               benchmark=benchmark_actual)
        assert qs1.cupy_enabled == True  # necesitamos GPU
        qs1.qc_matrix = None
        resultOK = qs1.qc_matrix_load()  # carga la matriz del circuito en la GPU
        assert resultOK
        qs1.instate_matrix = None
        resultOK = qs1.instate_matrix_load()
        assert resultOK
        resultOK = qs1.outstate_calculate()
        simu=qs1
        if simu.benchmark is not None:
            for corrida in range(1, corridasGPU + 1):
                print(f"INFO: benchmarking, corrida: {corrida}")
                _ = pd.DataFrame(columns=["clase", "corrida", "circuito", "num_qubits", "walltime"],
                                 data=[[simu.benchmark["clase"], corrida, qc1.name, qc1.num_qubits,
                                        simu.benchmark["walltime"]], ])
                display(_)
                benchmarks_corridos = pd.concat([benchmarks_corridos, _], ignore_index=True)
                resultOK = simu.outstate_calculate()

            benchmark_actual = None
        display(benchmarks_corridos)
        assert resultOK
