import matplotlib.pyplot as plt
from qiskit import *
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options, Sampler
from qiskit_ibm_runtime import RuntimeJob
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
        delay_periods = list(range(0, 402, 10))
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

        if len(gates) == 0:
            gate_name = ''
            target_qubit = 0
        else:
            drived_qubits = set()
            for gate in gates:
                if len(gate.qubits) != 1:
                    raise ValueError(
                        f'The gate is not a single-qubit gate:{gate}')
                drived_qubits.update(gate.qubits)
            if len(drived_qubits) != 1:
                raise ValueError(
                    'The gates were applied on more than on qubits.')
            gate = gates[0]
            gate_name = gate.operation.name
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

    def show_result(self, diff: 'CrosstalkInDelayTask | None' = None):
        if diff is not None:
            if self.backend != diff.backend:
                raise ValueError(
                    (f'Backend are not the same. '
                     f'self:{self.backend.name} diff:{diff.backend.name}'))
            if len(self.circuits) != len(diff.circuits):
                raise ValueError(
                    (f'The numbers of circuits are not equal. '
                     f'self:{len(self.circuits)} diff:{len(diff.circuits)}'))
            diff._get_result()

        self._get_result()
        figure = plt.figure(figsize=(12, 6))

        ax = plt.subplot2grid((8, 9), (0, 0), colspan=2, rowspan=3)
        ax.axis('off')
        ax.axis('tight')
        shots = self.job.inputs["run_options"]["shots"]
        resilience_level = self.job.inputs["resilience_settings"]["level"]
        plt.title((f'{self.backend.name}\n'
                   f'shots:{shots} resilience_level:{resilience_level}'),
                  loc='left')
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
            if diff is not None:
                probabilities = [
                    p - diff.result[qubit_index][i]
                    for i, p in enumerate(probabilities)
                ]
            plt.plot(self.delay_periods,
                     probabilities,
                     '-',
                     label=f'q_{qubit_index}')
        if diff is not None:
            plt.title('strength of crosstalk\n(this minus delay only)')
        else:
            plt.title('strength of crosstalk')
        plt.legend(loc='upper right')
        plt.xlabel('delay time (us)')
        plt.ylabel('probability of $| 1 \\rangle$')

        figure.tight_layout()
        plt.show()


if __name__ == '__main__':
    import argparse
    import configparser
    from pathlib import Path

    usage = f'''
  {Path(__file__).name} [-h] -m MACHINE [-g GATE] [-s SHOT]
  {Path(__file__).name} [-h] -j JOB_ID
'''
    description = ('Apply the given gate on a qubis and '
                   'show how other qubits evolve in time.')

    parser = argparse.ArgumentParser(description=description, usage=usage)

    new_task_group = parser.add_argument_group('New task')
    new_task_group.add_argument('-b',
                                '--backend',
                                type=str,
                                help='The name of the backend')
    new_task_group.add_argument('-g',
                                '--gate',
                                type=str,
                                default='',
                                help='The name of the gate will be applied')
    new_task_group.add_argument('-s',
                                '--shot',
                                dest='shot',
                                type=int,
                                default=100000,
                                help='The number of shot')

    retrieve_task_group = parser.add_argument_group('Retrieve task')
    retrieve_task_group.add_argument('-j',
                                     '--job_id',
                                     dest='job_id',
                                     type=str,
                                     help='The ID of the job of IMBQ runtime')
    general_group = parser.add_argument_group('General')
    general_group.add_argument(
        '-d',
        '--diff',
        dest='diff',
        type=str,
        help=(
            'The ID of the job that did not apply any gate. The number of the '
            'circuits and the backend should be the same as the input task.'))

    args = parser.parse_args()
    if len([s for s in (args.backend, args.job_id) if s is not None]) != 1:
        parser.print_help()
        exit(-1)

    target_qubit_mapping = {
        'ibm_perth': 1,
        'ibm_lagos': 1,
        'ibm_nairobi': 1,
        'ibmq_jakarta': 1,
        'ibmq_manila': 1,
        'ibmq_quito': 3,
        'ibmq_belem': 3,
        'ibmq_lima': 3,
        'ibmq_qasm_simulator': 1,
    }

    config = configparser.ConfigParser()
    config.read(Path(__file__).parent / 'config.ini')

    service = QiskitRuntimeService(channel='ibm_quantum',
                                   token=config['secret']['ibm_token'])

    diff_task = None
    if args.diff is not None:
        diff_job = service.job(args.diff)
        diff_task = CrosstalkInDelayTask.from_job(diff_job)
    if args.backend:
        options = Options(optimization_level=0,
                          resilience_level=0,
                          execution={'shots': args.shot})
        backend = service.backend(args.backend)
        target_qubit = target_qubit_mapping[args.backend]
        task = CrosstalkInDelayTask(backend, args.gate, target_qubit)

        with Session(service=service, backend=backend) as session:
            sampler = Sampler(session=session, options=options)
            task.run(sampler)
            task.show_result(diff_task)
            session.close()
    elif args.job_id:
        job = service.job(args.job_id)
        task = CrosstalkInDelayTask.from_job(job)
        task.show_result(diff_task)
    else:
        assert False
