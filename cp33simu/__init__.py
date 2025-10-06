import qiskit as qk
import qiskit_aer as aer
import numpy as np
from abc import ABC, abstractmethod

try:
    import cupy
except ImportError:
    print("WARN: cupy no está instalado, las simulaciones van a correr en CPU")
    import numpy as cupy
except Exception:
    raise

try:
    import mpi4py
    from mpi4py import MPI
except ImportError:
    print("WARN: mpi4py no está instalado")
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
        self.cupy_enabled = False
        if self.cupy_installed() and self.validate_cupy():
            #self.backend = aer.AerSimulator(method="statevector_gpu")
            #The qiskit-aer and qiskit-aer-gpu are mutually exclusive packages. They contain the same code except that the qiskit-aer-gpu package built with CUDA support enabled
            self.backend = aer.AerSimulator(method="statevector")
            self.cupy_enabled = True
        else:
            # print("WARN: ") ya sabe
            pass
        print(f"INFO: **{cupy.__name__}** funciona bien, v{cupy.__version__}")
        self.qc = qc
        print(f"INFO: Simulador creado: {self.qc.name}")
        self.num_qubits = self.qc.num_qubits
        print(f"INFO: Qbits  {self.num_qubits}")

        # inicializamos las matrices
        self.cupy_dtype=cupy.complex128 #TODO: ver otros tipitos
        self.qc_matrix=None
        self.instate_matrix=None
        self.outstate_matrix=None

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
        super().__init__(qc)
        self._GPUtopology = None  # atributo privado para topología GPU

    @property
    def GPUtopology(self):
        """
        Getter: configuración  GPU (grids, bloques, threads)

        :return: dict con configuración de topología GPU
        :rtype: dict or None
        """
        print(f"WARN: {type(self)}.GPUtopology es stub")
        return self._GPUtopology

    @GPUtopology.setter
    def GPUtopology(self, configuration):
        """
        Setter: Asigna la configuración de topología GPU

        :param configuration: Diccionario con configuración de topología (grids, bloques, etc)
        :type configuration: dict
        """
        self._GPUtopology = None
        topologyTemplate = {"grids":None,
                        "blocks":None,
                        "threads":None}
        try:
            if not isinstance(configuration, dict):
                msg=f"GPUtopologia debe ser un diccionario, con 'matrix' como clave, opcionalmente puede tener {list(topologyTemplate.keys())}"
                print(f"ERROR: {msg}")
                raise ValueError(msg)
            if not "matrix" in configuration.keys() or not isinstance(configuration["matrix"], cupy.ndarray):
                msg=f"valor 'matrix' debe estar y ser {cupy.ndarray.__name__}"
                print(f"ERROR: {msg}")
                raise ValueError(msg)
            if len(configuration["matrix"].shape) != 2:
                msg=f"matriz no es 2D: {configuration["matrix"].shape}"
                print(f"ERROR: {msg}")
                raise ValueError(msg)
            # todo: validar value["matrix"] comu matrix unitaria y la mar en coche de Schreadinger y su gato del orto
            # todo: avisar q ignoramos "grids","blocks","threads"
        except ValueError as e:
            self._GPUtopologia = None #si, otra vez
        except Exception:
            raise #rompé pepe
        else:
            # fecpeto-coorecto
            self._GPUtopology = dict(topologyTemplate, **configuration)

        if self._GPUtopology is not None:
            del configuration
            configuration=dict()
            with cupy.cuda.Device(0): #todo: soporte multi-device
                configuration["grids"]=cupy.cuda.runtime.getDeviceProperties(0)["multiProcessorCount"]
                configuration["blocks"]=cupy.cuda.runtime.getDeviceProperties(0)["maxThreadsPerBlock"]
                configuration["threads"]=cupy.cuda.runtime.getDeviceProperties(0)["maxThreadsPerMultiProcessor"]
                configuration["max_threads"]= configuration["grids"] * configuration["blocks"] * configuration["threads"]
                configuration["compute_capability"]=cupy.cuda.Device().compute_capability

            configuration["threads"] = 256# TODO: esto es harcodeado
            # https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Device.html#cupy.cuda.Device.compute_capability

            # sale con fritas
            self._GPUtopology = dict(self._GPUtopology, **configuration)


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
        from cupyx import jit
        # TODO: validar self.instate_matrix, self.qc_matrix y la mar en coche
        result=False #no-OK
        try:
            params = {"out": None}
            self.outstate_matrix = cupy.asnumpy(cupy.matmul(self.qc_matrix,self.instate_matrix), **params)

            print(f"INFO: estado de salida OK")
            print(f"DEBUG: estado de salida: {self.outstate_matrix}")
            result=True #VAMOOOO' lo' pibeeeeee'
        except Exception as e:
            print(f"ERROR: Falló al calcular el estado de salida: {e}")

        return result


class simuGPUbajo(simuGPU):
    """
    Simulador cuántico con en GPU utilizando CuPy.

    """

    def __init__(self, qc: qk.QuantumCircuit):
        """
        Inicializa el simulador con un circuito cuántico ingresado

        :param qc: circuito a simular
        :type qc: qk.QuantumCircuit
        """
        super().__init__(qc)
        from cupyx import jit
        self.jit = jit # serializate ésta, pickle :D
        print(f"INFO: Simulador GPU bajo creado: {self.qc.name} - Qbits  {self.num_qubits}")

    def validate_cupy(self):
        """
        Valida biblioteca CuPy, ejecuta una prueba simple, y realiza un benchmark liviano.
        wrapper de método de la clase ancestra

        :return: True sii CuPy está funcionando
        :rtype: bool
        """
        result = super().validate_cupy()#*args,**kwargs) ##siga-siga
        return result

    def cupy_installed(self):
        """
        Verifica si CuPy está instalado
        wrapper de método de la clase ancestra

        :return: True sii CuPy está disponible
        :rtype: bool
        """
        result = super().validate_cupy()  # *args,**kwargs) ##siga-siga
        return result

    def qc_matrix_load(self):
        """
        Carga la matriz del circuito.
        wrapper de método de la clase ancestra

        :return: True si la carga fue exitosa
        :rtype: bool
        """
        result = super().qc_matrix_load() ##siga-siga
        return result


    def instate_matrix_load(self, todos_ceros=True):
        """
        Carga la matriz del estado inicial del sistema cuántico.
        wrapper de método de la clase ancestra

        :param todos_ceros: Si el estado inicial debe ser |0...0>
        :type todos_ceros: bool
        :return: True si la carga fue exitosa
        :rtype: bool
        """
        result = super().instate_matrix_load()#*args,**kwargs) ##siga-siga
        return result


    def outstate_calculate(self):
        """
        Calcula la matriz del estado de salida.
        A diferencia de la clase ancestra implemente matmul a bajo nivel


        :return: Boolean cálculo salió bien
        :rtype: bool
        """
        # raise NotImplementedError()
        result = False
        try:
            if self.qc_matrix is None or self.instate_matrix is None:
                msg=f"deben cargarse antes qc_matrix Y  instate_matrix"
                raise ValueError(msg)
            self.GPUtopology = {"matrix":self.qc_matrix} #armo grids/bloques todo: stub
            if not self.GPUtopology is None:
                pass # TODO: ver qué hacemos si la dimension falla
            #if qc_matrix_dims[0] != instate_matrix_dims[1]:
            #    raise ValueError(f"ERROR: dimensiones de la matriz del circuito y del estado inicial no coinciden: {qc_matrix_dims} != {instate_matrix_dims}")
            #params = {"out": None}
            #self.outstate_matrix = cupy.asnumpy(cupy.matmul(self.qc_matrix, self.instate_matrix), **params)

            # Implementación con kernel de bajo nivel - "artesanal"

            @self.jit.rawkernel()
            def complex_matmul(qc_real, qc_imag, instate_real, instate_imag, outstate_real, outstate_imag, rows):
                """
                reimplementación de cupy.matmul() a bajo nivel.

                :param qc_real: 1D "row-major"
                :param qc_imag: 1D "row-major"
                :param instate_real: estado de entrada, parte real fila (no columna)
                :param instate_imag: estado de entrada, parte real fila (no columna)
                :param outstate_real: estado de salida, parte real fila (no columna), por referencia (se seobreescribe)
                :param outstate_imag: estado de salida, parte imag fila (no columna), por referencia (se seobreescribe)
                :param rows: lado de qc_matrix
                """
                result = False
                row = self.jit.blockIdx.x * self.jit.blockDim.x + self.jit.threadIdx.x

                if rows > self.GPUtopology["max_threads"]:
                    pass  #TODO: verificar q entra en la memoria ppal (o lo hacemos en el setter de GPUtopology?)

                if row < rows:
                    real_sum = 0.0
                    imag_sum = 0.0

                    for col in range(rows):
                        # arreglos separados real/imag
                        u_idx = row * rows + col
                        ur = qc_real[u_idx]
                        ui = qc_imag[u_idx]
                        sr = instate_real[col]
                        si = instate_imag[col]

                        # multiplicaión de complejos (ur + ui*j)(sr + si*j)
                        real_sum += ur * sr - ui * si
                        imag_sum += ur * si + ui * sr

                    outstate_real[row] = real_sum
                    outstate_imag[row] = imag_sum
                    result = True #TODO: ver errorlevel con jit

            # def unitary_by_state(unitary, state): no andubo. prepara datos \\
            # para el kernel cupy.ravel() https://docs.cupy.dev/en/stable/reference/generated/cupy.ravel.html
            state = self.instate_matrix.ravel()
            rows = state.shape[0]

            threads = self.GPUtopology["threads"] # era 256 hardodeado, igual OJO q GPUotpology es stub
            blocks = (rows + threads - 1) // threads # TODO: comparar con self.GPUtopology["blocks"]

            # separamos real/imaginario pero evitando ascontiguousarray, que no hubo forma
            u_real = self.qc_matrix.real.ravel()
            u_imag = self.qc_matrix.imag.ravel()
            s_real = state.real
            s_imag = state.imag

            out_real = cupy.empty(rows, dtype=cupy.float64) #COMPLEX128**(-2)
            out_imag = cupy.empty(rows, dtype=cupy.float64)

            # a ver si pegamos los tipos de dato q cuda entiende
            complex_matmul((cupy.uint32(blocks),), #ojo el casteo, timoteo
                           (cupy.uint32(threads),),
                           (u_real, u_imag, s_real, #ojo el casteo, timoteo
                            s_imag, out_real, out_imag, cupy.uint32(rows)))

            # volvemos a juntar
            self.outstate_matrix = cupy.asnumpy(out_real + 1j * out_imag)

            print(f"INFO: estado de salida OK (kernel CUDA bajo nivel)")
            print(f"DEBUG: estado de salida: {self.outstate_matrix}")
            print(f"\nINFO: estado de salida - amplitud de probabilidad de los {2 ** self.num_qubits} posibles estados:")
            for i, amp_proba in enumerate(self.outstate_matrix):  # cp33simu.cupy.asnumpy(outstate)):
                estado_bin = format(i, f'0{self.num_qubits}b')  # bin
                print(f"|{estado_bin}>: {amp_proba.real:.3f} + i{amp_proba.imag:.3f}")  # ya era real,pero molesta el j0.00000...
            result = True
        except ValueError as e:
            print(f"ERROR: {e}")
            result = False #no-OK
        except Exception as e:
            result = False #no-OK igual rompemos todo
            raise e
        return result #OK/no-OK


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
        if not "mpi4py" in globals():
            raise ImportError("ERROR: no se encuentra mpi4py. ver https://mpi4py.readthedocs.io/en/stable/install.html")
        print(f"INFO: Simulador MPI creado: {self.qc.name}")
        print(f"INFO: Qbits  {self.num_qubits}")

        self.mpi_rank = None
        self.mpi_size = None
        self.local_state_size = None
        self._mpi_dtype = None  # atributo privado para MPI datatype
        #raise NotImplementedError()

    @property
    def MPI_dtype(self):
        """
        Getter: Retorna el tipo de dato MPI correspondiente a cupy_dtype

        :return: Tipo de dato MPI
        :rtype: MPI.Datatype
        """
        if self._mpi_dtype is None:
            # TODO: implementar mapeo completo para otros tipos
            if self.cupy_dtype == cupy.complex128:
                self._mpi_dtype = MPI.COMPLEX16
            else:
                raise NotImplementedError(f"Mapeo MPI para {self.cupy_dtype} no implementado")
        return self._mpi_dtype

    @MPI_dtype.setter
    def MPI_dtype(self, value):
        """
        Setter: Asigna el tipo de dato MPI manualmente

        :param value: Tipo de dato MPI
        :type value: MPI.Datatype
        """
        self._mpi_dtype = value

    def qc_matrix_load(self):
        """
        Carga la matriz del circuito cuántico de forma distribuida usando MPI.

        :return: True si la carga fue exitosa, False en caso contrario
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




    def instate_matrix_load(self, todos_ceros=True):
        """
        Carga el estado inicial para distribuir entre los procesos MPI.

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
        Calcula el estado de salida de forma distribuida usando descomposición de dominio MPI.

        :return: True si cálculo fue exitoso
        :rtype: bool
        """
        # TODO: implementar cálculo distribuido
        print("INFO: outstate_calculate() - stub MPI")
        raise NotImplementedError("TODO: cálculo de estado de salida con MPI")
