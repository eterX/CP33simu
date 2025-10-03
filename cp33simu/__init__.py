import qiskit as qk
import qiskit_aer as aer
import numpy as np
from abc import ABC, abstractmethod

try:
    import cupy
except ImportError:
    print("WARN: cupy no está instalado, las simulaciones van a correr en CPU")
except Exception:
    raise

__all__ = ['simuGPU', 'simuGPUbajo', 'simuMPI']


class simuAbstracto(ABC):
    """
    Clase base, interfaz que todos los simuladores
    """

    def __init__(self, qc: qk.QuantumCircuit):
        """
        Inicializa el simulador con un circuito cuántico

        :param qc: circuito a simular
        :type qc: qk.QuantumCircuit
        """
        self.qc = qc
        self.num_qubits = self.qc.num_qubits
        self.qc_matrix = None
        self.instate_matrix = None
        self.outstate_matrix = None

    @abstractmethod
    def qc_matrix_load(self):
        """
        Carga la matriz del circuito.

        :return: True si la carga fue exitosa
        :rtype: bool
        """
        pass

    @abstractmethod
    def instate_matrix_load(self, todos_ceros=True):
        """
        Carga la matriz del estado inicial del sistema cuántico.

        :param todos_ceros: Si el estado inicial debe ser |0...0>
        :type todos_ceros: bool
        :return: True si la carga fue exitosa
        :rtype: bool
        """
        pass

    @abstractmethod
    def outstate_calculate(self):
        """
        Calcula el estado de salida aplicando el circuito al estado de entrada.

        :return: True si el cálculo fue exitoso
        :rtype: bool
        """
        pass


class simuGPU(simuAbstracto):
    #
    # Simulador del simulador de Qiskit, con GUP, cuentas de alto nivel (no kernels)
    #
    #
    #
    def __init__(self,qc: qk.QuantumCircuit):
        """
        Inicializa el simulador con un circuito cuántico ingresado

        :param qc: circuito a simualr
        :type qc: qk.QuantumCircuit
        """
        self.cupy_enabled = False
        if self.cupy_installed() and self.validate_cupy():
            #self.backend = aer.AerSimulator(method="statevector_gpu")
            #The qiskit-aer and qiskit-aer-gpu are mutually exclusive packages. They contain the same code except that the qiskit-aer-gpu package built with CUDA support enabled
            self.backend = aer.AerSimulator(method="statevector")
            self.cupy_enabled = True

        self.qc = qc
        print(f"INFO: Simulador creado: {self.qc.name}")
        self.num_qubits = self.qc.num_qubits
        print(f"INFO: Qbits  {self.num_qubits}")

        # inicializamos las matrices
        self.cupy_dtype=cupy.complex128 #TODO: ver otros tipitos
        self.qc_matrix=None
        self.instate_matrix=None
        self.outstate_matrix=None
        pass

    def validate_cupy(self):
        """
        Valida biblioteca CuPy, ejecuta una prueba simple, y realiza un benchmark liviano.

        :return: True sii CuPy está disponible y funcionando
        :rtype: bool
        """
        try:
            import cupy
            import time

            #print("INFO: Biblioteca CuPy - OK")
            #print(f"DEBUG: Versión de CuPy: {cupy.__version__}")

            # Prueba hola mundo
            test_array = cupy.array([1, 2, 3, 4, 5])
            print(f"INFO: Hola Mundo - Array: {test_array}")

            # Benchmark liviano: multiplicación de matrices
            size = 10000
            print(f"DEBUG: Ejecutando benchmark (multiplicación de matrices {size}x{size})...")

            a = cupy.random.rand(size, size, dtype=cupy.float32)
            b = cupy.random.rand(size, size, dtype=cupy.float32)

            # Calentamiento
            _ = cupy.dot(a, b)
            cupy.cuda.Stream.null.synchronize()

            # Benchmark
            start = time.perf_counter()
            c = cupy.dot(a, b)
            cupy.cuda.Stream.null.synchronize()
            end = time.perf_counter()

            elapsed_ms = (end - start) * 1000
            print(f"INFO: Benchmark completado en {elapsed_ms:.2f} ms")
            #print(f"INFO: GPU: {cupy.cuda.Device().name.decode()}")
            print(f"DEBUG: GPU.compute_capability: {cupy.cuda.Device().compute_capability}")
        except ImportError:
            print("WARN: No se encontró la biblioteca CuPy. Aceleración GPU no disponible.")
            return False
        except Exception as e:
            print(f"ERROR: La validación de CuPy falló: {e}")
            return False
        return True  # todo bien


    def cupy_installed(self):
        """

        :return: True sii CuPy está disponible y funcionando
        :rtype: bool
        """
        try:
            import cupy
            print(f"DEBUG: Versión de CuPy: {cupy.__version__}, GPU: {cupy.cuda.runtime.getDeviceProperties(0)["name"]}")
            #print(cupy.cuda.runtime.driverGetVersion())
            #print(cupy.cuda.runtime.runtimeGetVersion())
            size = 10
            a = cupy.random.rand(size, size, dtype=cupy.float32)

        except ImportError:
            print("WARN: No se encontró la biblioteca CuPy. Sin aceleración GPU")
            return False #perdón Niklaus
        except Exception as e:
            print(f"ERROR: falló la validación de CuPy:  cupy.random.rand() - > {e}")
            return False
        return True #todo bien

    def qc_matrix_load(self):
        """
        Carga la matriz del circuito en la GPU

        :raises Exception: si hay problemas cargando 
        :return: carga OK
        :rtype: bool
        """
        result = False # no-OK
        if self.qc_matrix is not None:
            print("ERROR: Matriz ya cargada. Para recargar, 'qc_matrix=None' y luego ejecutar 'qc_matrix_load()'")
        else:
            try:
                # https://docs.cupy.dev/en/stable/reference/generated/cupy.array.html#cupy.array
                params={'dtype':self.cupy_dtype}
                self.qc_matrix=cupy.array(qk.quantum_info.Operator.from_circuit(self.qc),
                                            **params)
                result = True
            except Exception as e:
                print(f"ERROR: Fallo al cargar la matriz del circuito: {e}")
            print(f"INFO: Estado inicial cargado: {self.qc_matrix}")
        return result

    def instate_matrix_load(self,todos_ceros=True):
        """
        Carga la estado inicial de los qubits en la GPU
        solo está soportao el estado |0...0>

        :param todos_ceros: estado inicial |0...0>.  Default a True
        :type todos_ceros: bool
        :return: carga OK
        :rtype: bool
        """
        result = False # no-OK
        if self.instate_matrix is not None:
            raise ValueError("ERROR: Matriz ya cargada. Para recargar, 'in_state_matrix=None' y luego ejecutar 'in_state_matrix_load()'")
        elif not todos_ceros:
            raise NotImplementedError("ERR: estado inicial distinto de |0...0> no implementado")

        try:
            # creamos el estado inicial
            #https://docs.cupy.dev/en/stable/reference/linalg.html
            #
            params={'dtype':self.cupy_dtype}
            self.instate_matrix = cupy.asarray([[1], [0]],
                                    **params)
            if self.num_qubits > 1:
                ket_cero = cupy.asarray([[1], [0]],
                                        **params)
                for _ in range(1,self.num_qubits):
                    # aplicamos https://docs.cupy.dev/en/stable/reference/generated/cupy.kron.html#cupy.kron
                    self.instate_matrix = cupy.kron(self.instate_matrix,ket_cero)
            else:
                print(f"WARN: para 1 qb, te vendo una HP48... baratita, baratita. Llamá 54-2600-HP48")
            params={'dtype':self.cupy_dtype}
            result = True
        except Exception as e:
            print(f"ERROR: Fallo al cargar  el estado inicial: {e}")
        print(f"INFO: Estado inicial cargado")
        print(f"DEBUG: Estado inicial: {self.instate_matrix}")
        return result #errorlevel

    def outstate_calculate(self):
        """
        Calcula la matriz del estado de salida

        :return: Boolean cálculo salió bien
        :rtype: bool
        """
        # TODO: validar self.instate_matrix, self.qc_matrix y la mar en coche
        result=False #no-OK
        try:
            params = {"out": None}
            self.outstate_matrix = cupy.asnumpy(cupy.matmul(self.qc_matrix,self.instate_matrix), #perdón Guido
                                            **params)
            print(f"INFO: estado de salida OK")
            print(f"DEBUG: estado de salida: {self.outstate_matrix}")
            result=True #VAMOOOO' lo' pibeeeeee'
        except Exception as e:
            print(f"ERROR: Falló al calcular el estado de salida: {e}")

        return result


class simuGPUbajo(simuGPU):
    """
    TODO: implementar métodos con GPU bajo nivel
    Todos los métodos dan NotImplementedError.
    """

    def __init__(self, qc: qk.QuantumCircuit):
        """
        Inicializa el simulador con un circuito cuántico ingresado

        :param qc: circuito a simular
        :type qc: qk.QuantumCircuit
        """
        raise NotImplementedError()

    def validate_cupy(self):
        """
        Valida biblioteca CuPy, ejecuta una prueba simple, y realiza un benchmark liviano.

        :return: True sii CuPy está funcionando
        :rtype: bool
        """
        raise NotImplementedError()

    def cupy_installed(self):
        """
        Verifica si CuPy está instalado

        :return: True sii CuPy está disponible
        :rtype: bool
        """
        raise NotImplementedError()

    def qc_matrix_load(self):
        """
        Carga la matriz del circuito.

        :return: True si la carga fue exitosa
        :rtype: bool
        """
        raise NotImplementedError()

    def instate_matrix_load(self, todos_ceros=True):
        """
        Carga la matriz del estado inicial del sistema cuántico.

        :param todos_ceros: Si el estado inicial debe ser |0...0>
        :type todos_ceros: bool
        :return: True si la carga fue exitosa
        :rtype: bool
        """
        raise NotImplementedError()

    def outstate_calculate(self):
        """
        Calcula la matriz del estado de salida

        :return: Boolean cálculo salió bien
        :rtype: bool
        """
        raise NotImplementedError()


class simuMPI(simuAbstracto):
    """
    Simulador cuántico distribuido usando MPI con descomposición de dominio.
    Implementación para la Sección 3 del proyecto.
    """

    def __init__(self, qc: qk.QuantumCircuit):
        """
        Inicializa el simulador MPI con un circuito cuántico

        :param qc: circuito a simular
        :type qc: qk.QuantumCircuit
        """
        super().__init__(qc)
        print(f"INFO: Simulador MPI creado: {self.qc.name}")
        print(f"INFO: Qbits  {self.num_qubits}")

        # TODO: inicializar MPI
        self.mpi_rank = None
        self.mpi_size = None
        self.local_state_size = None
        raise NotImplementedError()

    def qc_matrix_load(self):
        """
        Carga la matriz del circuito cuántico de forma distribuida usando MPI.

        :return: True si la carga fue exitosa, False en caso contrario
        :rtype: bool
        """
        # TODO: implementar carga distribuida
        print("INFO: qc_matrix_load() - stub MPI")
        raise NotImplementedError()

    def instate_matrix_load(self, todos_ceros=True):
        """
        Carga el estado inicial distribuido entre los procesos MPI.

        :param todos_ceros: Si el estado inicial debe ser |0...0>
        :type todos_ceros: bool
        :return: True si la carga fue exitosa, False en caso contrario
        :rtype: bool
        """
        # TODO: implementar carga de estado inicial distribuido
        print("INFO: instate_matrix_load() - stub MPI")
        raise NotImplementedError()

    def outstate_calculate(self):
        """
        Calcula el estado de salida de forma distribuida usando descomposición de dominio MPI.

        :return: True si cálculo fue exitoso
        :rtype: bool
        """
        # TODO: implementar cálculo distribuido
        print("INFO: outstate_calculate() - stub MPI")
        raise NotImplementedError("TODO: cálculo de estado de salida con MPI")
