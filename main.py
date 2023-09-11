import configparser
from pathlib import Path

import matplotlib.pyplot as plt
from qiskit import *
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options, Sampler, RuntimeJob
from qiskit.providers import BackendV1
from qiskit.compiler import transpile
from qiskit.circuit import CircuitInstruction
from qiskit.circuit import Barrier, Measure, Delay
from qiskit.visualization import plot_gate_map


class CrosstalkInDelayTask:
    '''
    A task to see how strengh the crosstalk is when there is no pulse applied
    on the qubit. The task drives a qubit with the given gate and waits a
    period of time before doing the measurement. The result shows the
    probability of being excited to the excited state on the qubits weren't
    drived by the given gate.
    '''

    def __init__(self, backend: BackendV1, gate_name: str,
                 target_qubit: int | None):
        '''
        Args:
            backend (Backend): An qiskit :class:`Backend` which the task will be
                executed on.
            gate_name (str): The name of the gate will drive the qubit at the
                begining.
        '''
        delay_periods = list(range(0, 402, 2))
        self._init_task(backend, gate_name, delay_periods, target_qubit)

    def _init_task(self,
                   backend: BackendV1,
                   gate_name: str,
                   delay_periods: list[float] | list[int],
                   target_qubit: int | None,
                   circuits: list[QuantumCircuit] | None = None):

        backend_configuration = backend.configuration()
        self.number_qubits = backend_configuration.n_qubits
        self.backend = backend
        self.target_qubit = target_qubit
        self.gate_name = gate_name
        self.delay_periods = delay_periods
        self.circuits = circuits
        self.job = None
        self.result = [
            [0.0] * len(self.delay_periods) for _ in range(self.number_qubits)
        ]

        if self.circuits is not None:
            return
        self.circuits = []
        for delay_us in self.delay_periods:
            circuit = QuantumCircuit(self.number_qubits)
            if gate_name != '':
                getattr(circuit, gate_name)(self.target_qubit)
            if delay_us > 0:
                circuit.delay(delay_us, self.target_qubit, 'us')
            circuit.measure_all()
            self.circuits.append(
                transpile(circuit,
                          backend=backend,
                          initial_layout=list(range(self.number_qubits)),
                          scheduling_method='alap'))

    def _update_result_from_quasi_dist(self, circuit_index: int,
                                       quasi_dist: dict[str, float]):
        for full_state, probability in quasi_dist.items():
            for qubit_index in range(self.number_qubits):
                mask = 1 << qubit_index
                if mask & full_state:
                    self.result[qubit_index][circuit_index] += probability

    def run(self, sampler: Sampler):
        self.job = sampler.run(self.circuits, skip_transpilation=True)
        print(f'>>> Job ID: {self.job.job_id()}')
        print(f'>>> Job Status: {self.job.status()}')

    def _get_result(self):
        for circuit_index, quasi_dist in enumerate(
                self.job.result().quasi_dists):
            self._update_result_from_quasi_dist(circuit_index, quasi_dist)

    @classmethod
    def from_job(cls, job: RuntimeJob):
        backend: BackendV1 = job.backend()
        circuits: list[QuantumCircuit] = job.inputs['circuits']

        instructions: list[CircuitInstruction] = circuits[-1].data
        gates = [
            i for i in instructions
            if type(i.operation) not in [Delay, Measure, Barrier]
        ]

        if len(gates) > 1:
            raise ValueError('Number of gates is not 1.')
        elif len(gates) == 0:
            gate_name = ''
            target_qubit = 0
        else:
            gate = gates[0]
            gate_name = gate.operation.name
            if len(gate.qubits) != 1:
                raise ValueError('The gate is not a single-qubit gate.')
            target_qubit = circuits[-1].find_bit(gate.qubits[0])[0]

        delay_periods = []
        dt_in_us = backend.configuration().to_dict()['dt'] * 1e-3
        for circuit in circuits:
            instructions: list[CircuitInstruction] = circuit.data
            for instruction in instructions:
                if type(instruction.operation) != Delay:
                    continue
                if circuit.find_bit(instruction.qubits[0])[0] != target_qubit:
                    continue
                delay_operation: Delay = instruction.operation
                delay_periods.append(delay_operation.duration * dt_in_us)
                break
            else:
                delay_periods.append(0)
        target_qubit = None if gate_name == '' else target_qubit

        task = cls.__new__(cls)
        task._init_task(backend, gate_name, delay_periods, target_qubit,
                        circuits)
        task.job = job
        return task

    def show_result(self):
        self._get_result()
        figure = plt.figure(figsize=(12, 6))

        ax = plt.subplot2grid((8, 9), (0, 0), colspan=2, rowspan=3)
        ax.axis('off')
        ax.axis('tight')
        plot_gate_map(self.backend, ax=ax)

        ax = plt.subplot2grid((8, 9), (0, 2), colspan=7, rowspan=3)
        rows = list(range(self.number_qubits))
        columns = ('frequency (GHz)', 'readout error', 'T1 (us)', 'T2 (us)',
                   'prob_meas0_prep1', 'prob_meas1_prep0')
        backend_properties = self.backend.properties()
        cell_text = []
        for i in range(self.number_qubits):
            qubit_property = backend_properties.qubit_property(i)
            cell_text.append([
                f'{qubit_property["frequency"][0] * 1e-9:.3f}',
                f'{qubit_property["readout_error"][0]:.4f}',
                f'{qubit_property["T1"][0] * 1e6:.2f}',
                f'{qubit_property["T2"][0] * 1e6:.2f}',
                f'{qubit_property["prob_meas0_prep1"][0]:.4f}',
                f'{qubit_property["prob_meas1_prep0"][0]:.4f}',
            ])
        ax.axis('off')
        ax.axis('tight')
        table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          colLabels=columns,
                          loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        ax = plt.subplot2grid((8, 9), (3, 0), colspan=6, rowspan=5)
        self.circuits[-1].draw('mpl', ax=ax)

        ax = plt.subplot2grid((8, 9), (3, 6), colspan=3, rowspan=5)
        for qubit_index, probabilities in enumerate(self.result):
            if qubit_index == self.target_qubit:
                plt.plot([], [], '-', label=f'q_{qubit_index}')
                continue
            plt.plot(self.delay_periods,
                     probabilities,
                     '-',
                     label=f'q_{qubit_index}')
        plt.title('strength of crosstalk')
        plt.legend(loc='upper right')
        plt.xlabel('delay time (us)')
        plt.ylabel('probability of $| 1 \\rangle$')

        figure.tight_layout()
        plt.show()


target_qubit_mapping = {
    'ibm_perth': 1,
    'ibm_lagos': 1,
    'ibm_nairobi': 1,
    'ibmq_jakarta': 1,
    'ibmq_manila': 1,
    'ibmq_quito': 3,
    'fake_quito': 3,
    'ibmq_belem': 3,
    'ibmq_lima': 3,
    'ibmq_qasm_simulator': 1,
}

config = configparser.ConfigParser()
config.read(Path(__file__).parent / 'config.ini')

device = config['device']['name']
options = Options(optimization_level=0,
                  resilience_level=0,
                  execution={'shots': 100000})
service = QiskitRuntimeService(channel='ibm_quantum',
                               token=config['secret']['ibm_token'])

task = CrosstalkInDelayTask.from_job(service.job('cjtj8vvtoe8ecf9ru6g0'))
task.show_result()
exit()

with Session(service=service, backend=device) as session:
    backend = service.backend(device)
    sampler = Sampler(session=session, options=options)
    tasks = []
    # for gate_name in ['x', 'h']:
    for gate_name in ['']:
        task = CrosstalkInDelayTask(backend, gate_name,
                                    target_qubit_mapping[device])
        task.circuits[-1].draw('mpl')
        task.run(sampler)
        tasks.append(task)

    # submit all jobs first then query the results
    for task in tasks:
        task.show_result()
    session.close()
