#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD.
# See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import re
import numpy as np
import logging
import json
import hashlib
import base64
from io import StringIO

from ase.cell import Cell
from ase.io import vasp

from nomad.units import ureg
from nomad.parsing.file_parser import TextParser, Quantity
from nomad.datamodel import EntryArchive
from runschema.run import Run, Program
from runschema.calculation import (
    Calculation,
    Energy,
    EnergyEntry,
    Forces,
    ForcesEntry,
    Stress,
    StressEntry,
    Thermodynamics,
    Dos,
    DosValues,
    BandStructure,
    BandEnergies,
)
from runschema.method import Method
from runschema.system import System, Atoms
from simulationworkflowschema import (
    Elastic,
    ElasticMethod,
    ElasticResults,
    Phonon,
    PhononMethod,
    PhononResults,
    Thermodynamics as WorkflowThermodynamics,
    ThermodynamicsResults,
)
from nomad.datamodel.metainfo.workflow import (
    Workflow,
    TaskReference,
    Link,
)
from electronicparsers.vasp import VASPParser

from .metainfo import aflow  # noqa


# Regex matching all AFLOW VASP output filenames:
#   vasprun.xml.relax1.xz, vasprun.xml.static.bz2, vasprun.xml.bands.xz, ...
AFLOW_VASPRUN_RE = re.compile(
    r'^vasprun\.xml\.(relax(\d+)|static(\d+)?|bands)\.(xz|bz2)$'
)

# Canonical ordered roles for every [VASP_RUN] directive type.
# 'relax1' is a placeholder - expanded to relax1..relaxN at runtime.
VASP_RUN_ROLES = {
    'GENERATE': [],
    'STATIC': ['static'],
    'STATIC_BANDS': ['static', 'bands'],
    'RELAX': ['relax1'],
    'RELAX_STATIC': ['relax1', 'static'],
    'RELAX_STATIC_BANDS': ['relax1', 'static', 'bands'],
}

# Modules with dedicated parse_* methods in AFLOWParser
AFLOW_MODULE_PARSERS = {'ael', 'agl', 'apl'}


def find_vasp_runs(maindir):
    """
    Scan *maindir* for ``vasprun.xml.<suffix>.xz/.bz2`` files.

    Returns an ``OrderedDict`` keyed by normalised role name, sorted in
    canonical AFLOW execution order: relax1 .. relaxN, static, bands.
    """
    found = {}
    for fname in sorted(os.listdir(maindir)):
        m = AFLOW_VASPRUN_RE.match(fname)
        if not m:
            continue
        suffix = m.group(1)  # e.g. 'relax1', 'static', 'bands'
        role = re.sub(r'^static\d+$', 'static', suffix)  # 'static2' > 'static'
        if role not in found:
            found[role] = os.path.join(maindir, fname)

    def _sort_key(role):
        rm = re.match(r'^relax(\d+)$', role)
        if rm:
            return (0, int(rm.group(1)))
        return {'static': (1, 0), 'bands': (2, 0)}.get(role, (3, 0))

    return dict(sorted(found.items(), key=lambda kv: _sort_key(kv[0])))


def parse_vasp_run_directive(vasp_run_str):
    """
    Parse the raw ``[VASP_RUN]`` value from *aflow.in*.

    Examples::

        'RELAX_STATIC_BANDS=2'  >  ('RELAX_STATIC_BANDS', 2)
        'STATIC_BANDS'          >  ('STATIC_BANDS', 0)
        'RELAX=3'               >  ('RELAX', 3)

    Returns ``(workflow_type, n_relax)`` where *n_relax* is the explicit
    integer after ``=``, defaulting to 2 for any RELAX variant.
    """
    if not vasp_run_str:
        return None, 0
    parts = vasp_run_str.strip().split('=')
    workflow_type = parts[0].strip()
    if len(parts) > 1:
        try:
            n_relax = int(parts[1].strip())
        except ValueError:
            n_relax = 2
    else:
        n_relax = 2 if 'RELAX' in workflow_type else 0
    return workflow_type, n_relax


def expected_roles_from_directive(workflow_type, n_relax):
    """
    Build the ordered list of expected VASP run roles.

    Expands the ``'relax1'`` placeholder in :data:`VASP_RUN_ROLES` into
    ``['relax1', 'relax2', .., 'relaxN']`` according to *n_relax*.
    """
    base = list(VASP_RUN_ROLES.get(workflow_type, []))
    if 'relax1' in base and n_relax > 1:
        relax_roles = [f'relax{i}' for i in range(1, n_relax + 1)]
        idx = base.index('relax1')
        base = base[:idx] + relax_roles + base[idx + 1 :]
    return base


def infer_workflow_type(roles):
    """
    Infer the AFLOW workflow type from whichever roles were actually found
    on disk.  Used as a fallback when ``[VASP_RUN]`` is absent.
    """
    has_relax = any(re.match(r'^relax\d+$', r) for r in roles)
    has_static = 'static' in roles
    has_bands = 'bands' in roles

    if has_relax and has_static and has_bands:
        return 'RELAX_STATIC_BANDS'
    if has_relax and has_static:
        return 'RELAX_STATIC'
    if has_static and has_bands:
        return 'STATIC_BANDS'
    if has_relax:
        return 'RELAX'
    if has_static:
        return 'STATIC'
    return 'UNKNOWN'


def parse_vasp_archive(filepath, logger):
    """
    Run the NOMAD VASP parser on *filepath* and return the child
    :class:`EntryArchive`.  Returns ``None`` on failure.
    """
    child = EntryArchive()
    try:
        VASPParser().parse(filepath, child, logger)
    except Exception as exc:
        logger.warning(
            'Could not parse VASP file',
            filepath=filepath,
            exc_info=exc,
        )
        return None
    return child


class AflowOutParser(TextParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_quantities(self):
        def str_to_property(val_in):
            val = val_in.split('=')
            return val[0].strip().replace(' ', '_').lower(), val[-1].split('//')[
                0
            ].strip()

        self._quantities = [
            Quantity(
                'property',
                r'\n *\[(.+)\](.+?=.+)',
                str_operation=str_to_property,
                repeats=True,
            ),
            Quantity(
                'section',
                r'(\[.+?\]START[\s\S]+?\]STOP)',
                repeats=True,
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'name',
                            r'\[(.+)\]START',
                            str_operation=lambda x: x.lower(),
                            dtype=str,
                        ),
                        Quantity('key_value', r'\n *([^#]\S+)=(\S+)', repeats=True),
                        Quantity(
                            'array',
                            rf'\n *(\d[\s\S]+?\d\s*)\[.+?STOP',
                            dtype=np.dtype(np.float64),
                        ),
                    ]
                ),
            ),
        ]

    def parse(self, key=None):
        super().parse(key)
        for property in self._results.get('property', []):
            self._results[property[0]] = property[1]
        for section in self._results.get('section', []):
            if section.key_value is not None:
                result = dict()
                for k, v in section.get('key_value', []):
                    result[k] = v
                self._results[section.name] = result
            elif section.array is not None:
                self._results[section.name] = section.array


class AflowInParser(AflowOutParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_quantities(self):
        super().init_quantities()
        self._quantities += [
            Quantity('aflow_version', r'Stefano Curtarolo \- \(AFLOW V([\d\.]+)\)'),
            Quantity(
                'vasp_run',
                r'\[VASP_RUN\](\S+)',
                dtype=str,
                convert=False,
            ),
            Quantity(
                'poscar',
                r'\[VASP_POSCAR_MODE_EXPLICIT\]START\s*([\s\S]+?)\[VASP_POSCAR_MODE_EXPLICIT\]STOP',
                str_operation=lambda x: x,
                convert=False,
                repeats=True,
            ),
            Quantity(
                'aflow_composition',
                r'\[AFLOW\] COMPOSITION=(\S+)',
                sub_parser=TextParser(
                    quantities=[
                        Quantity('species', r'([A-Z][a-z]*)', repeats=True, dtype=str),
                        Quantity(
                            'composition',
                            r'(\d+)\|',
                            repeats=True,
                            dtype=np.dtype(np.int32),
                        ),
                    ]
                ),
            ),
        ] + [
            Quantity(
                module.lower(),
                r'\n *\[AFLOW\_%s\]CALC([\s\S]+?)\[AFLOW\] \*' % module,
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'parameters',
                            r'\[AFLOW\_%s\](.+?)=(\S+)' % module,
                            repeats=True,
                        )
                    ]
                ),
            )
            for module in ['AEL', 'AGL', 'APL', 'QHA', 'AAPL']
        ]

    def parse(self, key=None):
        super().parse(key)

        if (
            self._results.get('poscar') is not None
            and self._results.get('geometry') is None
        ):
            try:
                atoms = vasp.read_vasp(StringIO(self._results['poscar'][-1]))
                self._results['cell'] = atoms.get_cell()
                self._results['geometry'] = atoms.get_cell().cellpar()
                composition = self._results['aflow_composition']
                self._results['species'] = composition.species
                self._results['composition'] = [
                    int(c) for c in composition._results['composition']
                ]
                self._results['positions_cartesian'] = atoms.get_positions()
            except Exception:
                pass

        if self._results.get('loop') is None:
            self._results['loop'] = [
                module
                for module in ['ael', 'agl', 'apl', 'qha', 'aapl']
                if module in self._results
            ]


class AFLOWParser:
    def __init__(self):
        self.ael_parser = AflowOutParser()
        self.agl_parser = AflowOutParser()
        self.apl_parser = AflowOutParser()
        self.aflowin_parser = AflowInParser()

        self._metainfo_map = {
            'stiffness_tensor': 'elastic_constants_matrix_second_order',
            'compliance_tensor': 'compliance_matrix_second_order',
            'poisson_ratio': 'poisson_ratio_hill',
            'bulk_modulus_vrh': 'bulk_modulus_hill',
            'shear_modulus_vrh': 'shear_modulus_hill',
            'youngs_modulus_vrh': 'Young_modulus_hill',
            'pughs_modulus_ratio': 'pugh_ratio_hill',
            'applied_pressure': 'x_aflow_ael_applied_pressure',
            'average_external_pressure': 'x_aflow_ael_average_external_pressure',
        }

    def init_parser(self):
        if '.json' in self.filepath:
            self.aflow_data = json.load(open(self.filepath))
        else:
            self.aflowin_parser.mainfile = self.filepath
            self.aflow_data = self.aflowin_parser

    def get_aflow_file(self, filename):
        files = [f for f in os.listdir(self.maindir) if filename in f]
        if not files:
            files = ['']
        return os.path.join(self.maindir, files[0])

    def parse_structures(self, module):
        try:
            structures = json.load(
                open(os.path.join(self.maindir, '%s_energy_structures.json' % module))
            ).get('%s_energy_structures' % module, [])
        except Exception:
            structures = []

        for structure in structures:
            sec_calc = Calculation()
            self.archive.run[-1].calculation.append(sec_calc)
            sec_thermo = Thermodynamics()
            sec_calc.thermodynamics.append(sec_thermo)
            if structure.get('energy') is not None:
                sec_calc.energy = Energy(
                    total=EnergyEntry(value=structure.get('energy') * ureg.eV)
                )
            if structure.get('pressure') is not None:
                sec_thermo.pressure = structure.get('pressure') * ureg.kbar
            if structure.get('stress_tensor') is not None:
                sec_calc.stress = Stress(
                    total=StressEntry(value=structure.get('stress_tensor') * ureg.kbar)
                )
            if structure.get('structure') is not None:
                sec_system = System()
                self.archive.run[-1].system.append(sec_system)
                sec_system.atoms = Atoms()
                struc = structure.get('structure')
                sec_system.atoms.labels = [
                    atom.get('name') for atom in struc.get('atoms', [])
                ]
                sec_system.atoms.concentrations = [
                    atom.get('occupancy') for atom in struc.get('atoms', [])
                ]
                if struc.get('lattice') is not None:
                    sec_system.atoms.lattice_vectors = (
                        struc.get('lattice') * ureg.angstrom * struc.get('scale', 1)
                    )
                positions = [atom.get('position') for atom in struc.get('atoms', [])]
                if struc.get('coordinates_type', 'direct').lower().startswith('d'):
                    if sec_system.atoms.lattice_vectors is not None:
                        positions = np.dot(
                            positions, sec_system.atoms.lattice_vectors.magnitude
                        ) * sec_system.atoms.lattice_vectors.units
                sec_system.atoms.positions = positions

    def parse_agl(self):
        sec_run = Run()
        self.archive.run.append(sec_run)
        sec_run.program = Program(
            name='AFlow', version=self.aflow_data.get('aflow_version', 'unknown')
        )

        self.parse_structures('AGL')

        self.agl_parser.mainfile = self.get_aflow_file('aflow.agl.out')
        thermal_properties = self.agl_parser.get('agl_thermal_properties_temperature')
        if thermal_properties is None:
            return

        workflow = WorkflowThermodynamics(results=ThermodynamicsResults())

        thermal_properties = np.reshape(
            thermal_properties, (len(thermal_properties) // 9, 9)
        )
        thermal_properties = np.transpose(thermal_properties)
        energies = self.agl_parser.get('agl_energies_temperature')
        energies = np.reshape(energies, (len(energies) // 9, 9))
        energies = np.transpose(energies)

        workflow.results.temperature = thermal_properties[0] * ureg.K
        workflow.results.gibbs_free_energy = energies[1] * ureg.eV
        workflow.results.vibrational_free_energy = energies[2] * ureg.meV
        workflow.results.vibrational_internal_energy = energies[3] * ureg.meV
        workflow.results.vibrational_entropy = energies[4] * ureg.meV / ureg.K
        workflow.results.heat_capacity_c_v = (
            thermal_properties[4] * ureg.boltzmann_constant
        )
        workflow.results.heat_capacity_c_p = (
            thermal_properties[5] * ureg.boltzmann_constant
        )
        # TODO add these to metainfo def
        # workflow.results.thermal_conductivity = thermal_properties[1] * ureg.watt / ureg.m * ureg.K
        # sec_debye.debye_temperature = thermal_properties[2] * ureg.K
        # sec_debye.gruneisen_parameter = thermal_properties[3]
        # sec_debye.thermal_expansion = thermal_properties[6] / ureg.K
        # sec_debye.bulk_modulus_static = thermal_properties[7] * ureg.GPa
        # sec_debye.bulk_modulus_isothermal = thermal_properties[8] * ureg.GPa
        self.archive.workflow2 = workflow

    def parse_ael(self):
        sec_run = Run()
        self.archive.run.append(sec_run)
        sec_run.program = Program(
            name='AFlow', version=self.aflow_data.get('aflow_version', 'unknown')
        )

        self.parse_structures('AEL')

        self.ael_parser.mainfile = self.get_aflow_file('aflow.ael.out')
        workflow = Elastic(method=ElasticMethod(), results=ElasticResults())
        workflow.method.energy_stress_calculator = 'vasp'
        workflow.method.calculation_method = 'stress'
        workflow.method.elastic_constants_order = 2

        paths = [
            d for d in self.aflow_data.get('files', []) if d.startswith('ARUN.AEL')
        ]
        deforms = np.array(
            [d.split('_')[-2:] for d in paths], dtype=np.dtype(np.float64)
        )
        strains = [d[1] for d in deforms if d[0] == 1]
        workflow.results.n_deformations = int(max(np.transpose(deforms)[0]))
        workflow.results.n_strains = len(strains)
        workflow.method.strain_maximum = max(strains) - 1.0

        for key, val in self.ael_parser.get('ael_results', {}).items():
            key = key.replace('ael_', '')
            key = self._metainfo_map.get(key, key)
            if 'modulus' in key or 'pressure' in key:
                val = val * ureg.GPa
            elif 'speed' in key:
                val = val * (ureg.m / ureg.s)
            elif 'temperature' in key:
                val = val * ureg.K
            setattr(workflow.results, key, val)

        if self.ael_parser.ael_stiffness_tensor is not None:
            workflow.results.elastic_constants_matrix_second_order = (
                np.reshape(self.ael_parser.ael_stiffness_tensor, (6, 6)) * ureg.GPa
            )

        if self.ael_parser.ael_compliance_tensor is not None:
            workflow.results.compliance_matrix_second_order = np.reshape(
                self.ael_parser.ael_compliance_tensor, (6, 6)
            )

        self.archive.workflow2 = workflow

    def parse_apl(self):
        sec_run = Run()
        self.archive.run.append(sec_run)
        sec_run.program = Program(
            name='AFlow', version=self.aflow_data.get('aflow_version', 'unknown')
        )
        sec_scc = Calculation()
        sec_run.calculation.append(sec_scc)

        try:
            dos = np.transpose(
                np.loadtxt(self.get_aflow_file('flow.apl.phonon_dos.out.xz'))
            )
        except Exception:
            dos = None

        if dos is not None:
            sec_dos = Dos()
            sec_scc.dos_phonon.append(sec_dos)
            sec_dos.energies = dos[2] * ureg.millielectron_volt
            sec_dos.total.append(
                DosValues(value=dos[3] * (1 / ureg.millielectron_volt))
            )

        try:
            kpoints = np.transpose(
                np.loadtxt(self.get_aflow_file('aflow.apl.hskpoints.out.xz'))
            )
            n_kpoints = int(max(kpoints[3])) + 1
            kpoints = kpoints[:3]
            kpoints = np.reshape(kpoints, (3, len(kpoints[0]) // n_kpoints, n_kpoints))
            kpoints = np.transpose(kpoints, axes=(1, 2, 0))

            bandstructure = np.transpose(
                np.loadtxt(self.get_aflow_file('aflow.apl.phonon_dispersion.out.xz'))
            )
            bandstructure = bandstructure[2:]
            bandstructure = np.reshape(
                bandstructure,
                (len(bandstructure), len(bandstructure[0]) // n_kpoints, n_kpoints),
            )
            bandstructure = np.transpose(bandstructure, axes=(1, 2, 0))
        except Exception:
            kpoints = None

        if kpoints is not None:
            sec_bandstructure = BandStructure()
            sec_scc.band_structure_phonon.append(sec_bandstructure)
            for n_segment in range(len(kpoints)):
                sec_segment = BandEnergies()
                sec_bandstructure.segment.append(sec_segment)
                sec_segment.kpoints = kpoints[n_segment]
                sec_segment.energies = (
                    np.reshape(
                        bandstructure[n_segment],
                        (1, *np.shape(bandstructure[n_segment])),
                    )
                    * ureg.millielectron_volt
                )

        self.apl_parser.mainfile = self.get_aflow_file(
            'aflow.apl.thermodynamic_properties.out'
        )

        workflow = Phonon(method=PhononMethod(), results=PhononResults())

        workflow.method.force_calculator = 'vasp'
        mesh = self.aflowin_parser.get('aflow_apl_dosmesh')
        if mesh is not None:
            try:
                cell = Cell.fromcellpar(self.aflowin_parser.geometry)
                workflow.method.mesh_density = (
                    np.product([int(m) for m in mesh.split('x')]) / cell.volume
                )
            except Exception:
                pass

        self.apl_parser.mainfile = self.get_aflow_file('aflow.apl.group_velocities.out')
        group_velocity = self.apl_parser.get('apl_group_velocity')
        if group_velocity is not None:
            try:
                qpoints = self.apl_parser.apl_qpoints
                qpoints = np.reshape(qpoints, (len(qpoints) // 4, 4))
                group_velocity = np.reshape(
                    group_velocity, (len(qpoints), len(group_velocity) // len(qpoints))
                )
                group_velocity = np.transpose(np.transpose(group_velocity)[1:])
                workflow.results.qpoints = np.transpose(np.transpose(qpoints)[1:])
                workflow.results.group_velocity = (
                    np.reshape(
                        group_velocity,
                        (len(group_velocity), len(group_velocity[0]) // 3, 3),
                    )
                    * ureg.kilometer
                    / ureg.second
                )
            except Exception:
                pass

        self.apl_parser.mainfile = self.get_aflow_file(
            'aflow.apl.thermodynamic_properties.out.xz'
        )
        apl_thermo = self.apl_parser.get('apl_thermo')
        # TODO handle multiple workflows
        if apl_thermo is not None:
            apl_thermo = np.transpose(np.reshape(apl_thermo, (len(apl_thermo) // 6, 6)))
            sec_thermo = WorkflowThermodynamics(results=ThermodynamicsResults())
            sec_thermo.results.temperature = apl_thermo[0] * ureg.kelvin
            sec_thermo.results.internal_energy = apl_thermo[2] * ureg.millielectron_volt
            sec_thermo.results.helmholtz_free_energy = (
                apl_thermo[3] * ureg.millielectron_volt
            )
            sec_thermo.results.entropy = apl_thermo[4] * ureg.boltzmann_constant
            sec_thermo.results.heat_capacity_c_v = (
                apl_thermo[5] * ureg.boltzmann_constant
            )

        self.archive.workflow2 = workflow

        # TODO parse systems for each displacements

        # TODO parse displacements, force constants, dynamical matrix


    # ---- VASP run workflow helpers ----

    def _combine_dos_and_bands(self, roles, child_archives):
        """
        Create a combined DOS + band structure view on the aflow.in entry
        itself, without touching either VASP child archive.

        A new Calculation section is appended to archive.run[0] containing:
          - band_structure_electronic copied from the bands run
          - dos_electronic copied from the static (or last relax) run

        NOMAD's normalizer detects both quantities in the same calculation and
        renders them together in the electronic structure viewer on the
        aflow.in overview page. Neither VASP entry is modified.
        """
        if 'bands' not in child_archives:
            return

        # Collect band structure from bands run
        bands_child = child_archives['bands']
        bs_list = []
        if bands_child.run and bands_child.run[-1].calculation:
            bs_list = bands_child.run[-1].calculation[-1].band_structure_electronic

        if not bs_list:
            self.logger.warning(
                'No band_structure_electronic in bands run - '
                'skipping combined plot on aflow.in entry'
            )
            return

        dos_source = child_archives.get('static')
        dos_list = []
        if dos_source is not None and dos_source.run and dos_source.run[-1].calculation:
            dos_list = dos_source.run[-1].calculation[-1].dos_electronic

        if not dos_list:
            self.logger.info(
                'No electronic DOS found - band structure only will be added '
                'to aflow.in entry'
            )

        dos_role = next(
            (r for r in roles if child_archives.get(r) is dos_source),
            'none',
        )

        # Append a new combined Calculation to archive.run[0].
        # run[0] is the aflow.in run created in parse(). We append a fresh
        # Calculation rather than modifying the existing one (which holds
        # energy, forces, etc. parsed from aflow.in itself).
        sec_combined = Calculation()
        self.archive.run[0].calculation.append(sec_combined)

        for bs in bs_list:
            sec_combined.band_structure_electronic.append(bs)

        for dos in dos_list:
            sec_combined.dos_electronic.append(dos)

        self.logger.info(
            'Created combined DOS+bands Calculation on aflow.in entry',
            dos_source_role=dos_role,
            n_dos=len(dos_list),
            n_bs=len(bs_list),
        )

    def _entry_id_for(self, mainfile_rel):
        """
        Compute the NOMAD entry ID for a mainfile given its path relative to
        the upload root.  Replicates NOMAD's internal hash: first 28 chars of
        URL-safe base64(SHA-512(upload_id + mainfile_path)).
        """
        upload_id = getattr(self.archive.m_context, 'upload_id', None)
        # for local runs check *here*
        if upload_id is None:
            upload_id = os.path.dirname(mainfile_rel)
        raw = hashlib.sha512((upload_id + mainfile_rel).encode()).digest()
        return base64.urlsafe_b64encode(raw).decode().rstrip('=')[:28]

    def _ref(self, entry_id, path):
        """Return a NOMAD archive reference string for a given entry and path."""
        return f'/entries/{entry_id}/archive#{path}'

    def _has_section(self, child, path):
        """
        Check whether *path* (e.g. '/run/0/system/0') resolves to a
        non-empty section in *child*.  Avoids building references to
        sections that don't exist.
        """
        try:
            parts = [p for p in path.strip('/').split('/') if p]
            obj = child
            for part in parts:
                if part.lstrip('-').isdigit():
                    obj = obj[int(part)]
                else:
                    obj = getattr(obj, part)
            return obj is not None
        except Exception:
            return False

    def _last_system_index(self, child):
        """Return the index of the last system in child.run[-1].system."""
        try:
            return len(child.run[-1].system) - 1
        except Exception:
            return -1

    def _last_calc_index(self, child):
        """Return the index of the last calculation in child.run[-1].calculation."""
        try:
            return len(child.run[-1].calculation) - 1
        except Exception:
            return -1

    def _build_vasp_workflow(
        self,
        workflow_type,
        roles,
        vasp_runs,
        child_archives,
        upload_prefix,
    ):
        """
        Construct a workflow2 on the aflow.in archive using string-path
        references to the individual VASP entries.

        All section references are built as ``/entries/<id>/archive#<path>``
        strings so NOMAD resolves them via its normal reference mechanism
        rather than receiving live Python objects (which cause the
        ``qualified_name`` AttributeError).
        """
        workflow = Workflow(name=f'AFLOW {workflow_type} Workflow')

        # Pre-compute entry IDs for all roles that have a file on disk
        entry_ids = {}
        for role, filepath in vasp_runs.items():
            fname = os.path.basename(filepath)
            rel_path = os.path.join(upload_prefix, fname) if upload_prefix else fname
            eid = self._entry_id_for(rel_path)
            if eid is not None:
                entry_ids[role] = eid
            else:
                self.logger.warning(
                    f'Could not compute entry ID for role "{role}" '
                    f'(upload_id not available in context) - '
                    f'workflow references for this role will be skipped'
                )

        tasks = []
        prev_role = None

        for role in roles:
            eid = entry_ids.get(role)
            child = child_archives.get(role)
            if eid is None or child is None:
                prev_role = role
                continue

            task = TaskReference()
            task.name = role.upper()

            # inputs: structure from previous run, or own first system
            if prev_role and entry_ids.get(prev_role) and child_archives.get(prev_role):
                prev_eid = entry_ids[prev_role]
                prev_child = child_archives[prev_role]
                last_sys = self._last_system_index(prev_child)
                task.inputs = [
                    Link(
                        name=f'Structure from {prev_role.upper()}',
                        section=self._ref(prev_eid, f'/run/0/system/{last_sys}'),
                    )
                ]
            else:
                if self._has_section(child, '/run/0/system/0'):
                    task.inputs = [
                        Link(
                            name='Input structure',
                            section=self._ref(eid, '/run/0/system/0'),
                        )
                    ]

            # outputs: last calculation
            last_calc = self._last_calc_index(child)
            if last_calc >= 0:
                task.outputs = [
                    Link(
                        name=f'{role.upper()} result',
                        section=self._ref(eid, f'/run/0/calculation/{last_calc}'),
                    )
                ]

            # task reference: prefer workflow2, fall back to calculation
            if child.workflow2 is not None:
                task.task = self._ref(eid, '/workflow2')
            elif last_calc >= 0:
                task.task = self._ref(eid, f'/run/0/calculation/{last_calc}')

            tasks.append(task)
            prev_role = role

        workflow.tasks = tasks

        # Global workflow inputs / outputs
        first_role = next(
            (r for r in roles if r in entry_ids and r in child_archives), None
        )
        last_role = next(
            (r for r in reversed(roles) if r in entry_ids and r in child_archives), None
        )

        if first_role:
            workflow.inputs = [
                Link(
                    name='Input structure',
                    section=self._ref(entry_ids[first_role], '/run/0/system/0'),
                )
            ]
        if last_role:
            last_calc = self._last_calc_index(child_archives[last_role])
            if last_calc >= 0:
                workflow.outputs = [
                    Link(
                        name='Final result',
                        section=self._ref(
                            entry_ids[last_role],
                            f'/run/0/calculation/{last_calc}',
                        ),
                    )
                ]

        self.archive.workflow2 = workflow

    # Top-level VASP run orchestration

    def parse_vasp_runs(self):
        """
        1. Read ``[VASP_RUN]`` from *aflow.in* to determine the workflow type
           and expected roles.
        2. Find actual ``vasprun.xml.*.xz`` files on disk and warn about
           any mismatch with the directive.
        3. Parse each VASP file into a child :class:`EntryArchive`.
        4. Combine DOS from the static (or last relax) run into the bands
           calculation so NOMAD can render them together.
        5. Build a ``workflow2`` on the ``aflow.in`` entry linking all runs
           using string-path references (``/entries/<id>/archive#<path>``).
        """
        vasp_run_str = self.aflow_data.get('vasp_run')
        workflow_type, n_relax = parse_vasp_run_directive(vasp_run_str)

        vasp_runs = find_vasp_runs(self.maindir)

        if not vasp_runs:
            self.logger.warning('No vasprun.xml.*.xz files found alongside aflow.in')
            return

        if workflow_type is not None:
            expected = expected_roles_from_directive(workflow_type, n_relax)
        else:
            self.logger.warning(
                '[VASP_RUN] directive absent from aflow.in - '
                'inferring workflow type from files on disk'
            )
            expected = list(vasp_runs.keys())
            workflow_type = infer_workflow_type(expected)
            n_relax = sum(1 for r in expected if re.match(r'^relax\d+$', r))

        for role in expected:
            if role not in vasp_runs:
                self.logger.warning(
                    f'Expected role "{role}" from [VASP_RUN]={vasp_run_str} '
                    f'but no matching file found on disk'
                )

        # Include unexpected files (e.g. restarted/extended runs)
        roles = list(expected)
        for role in vasp_runs:
            if role not in roles:
                self.logger.info(
                    f'Unexpected VASP run "{role}" found on disk '
                    f'(not in [VASP_RUN]={vasp_run_str}) - including anyway'
                )
                roles.append(role)

        # Determine the upload-relative prefix for this directory
        # (needed to reproduce NOMAD's mainfile path for entry ID hashing)
        try:
            upload_root = self.archive.m_context.raw_path()
            upload_prefix = os.path.relpath(self.maindir, upload_root)
            if upload_prefix == '.':
                upload_prefix = ''
        except Exception:
            upload_prefix = ''

        self.logger.info(
            'AFLOW VASP workflow configuration',
            workflow_type=workflow_type,
            n_relax=n_relax,
            roles=roles,
            upload_prefix=upload_prefix,
        )

        child_archives = {}
        for role in roles:
            filepath = vasp_runs.get(role)
            if filepath is None:
                continue
            self.logger.info(f'Parsing VASP run "{role}"', filepath=filepath)
            child = parse_vasp_archive(filepath, self.logger)
            if child is not None:
                child_archives[role] = child

        if not child_archives:
            self.logger.warning('No VASP runs could be parsed')
            return

        if 'bands' in child_archives and len(child_archives) > 1:
            self._combine_dos_and_bands(roles, child_archives)

        self._build_vasp_workflow(
            workflow_type,
            roles,
            vasp_runs,
            child_archives,
            upload_prefix,
        )

    def parse(self, filepath, archive, logger):
        self.filepath = os.path.abspath(filepath)
        self.archive = archive
        self.maindir = os.path.dirname(self.filepath)
        self.logger = logger if logger is not None else logging

        self.init_parser()

        sec_run = Run()
        self.archive.run.append(sec_run)
        sec_run.program = Program(
            name='AFlow', version=self.aflow_data.get('aflow_version', 'unknown')
        )

        # parse run metadata
        run_quantities = ['aurl', 'auid', 'data_api', 'data_source', 'loop']
        for key in run_quantities:
            val = self.aflow_data.get(key)
            if val is not None:
                setattr(sec_run, 'x_aflow_%s' % key, val)

        # System (structure)
        sec_system = System()
        sec_run.system.append(sec_system)
        sec_system.atoms = Atoms()
        lattice_parameters = self.aflow_data.get('geometry')
        if lattice_parameters is not None:
            cell = self.aflow_data.get('cell', Cell.fromcellpar(lattice_parameters))
            sec_system.atoms.lattice_vectors = cell.array * ureg.angstrom
            sec_system.atoms.periodic = [True, True, True]
        species = self.aflow_data.get('species', [])
        atom_labels = []
        for n, specie in enumerate(species):
            atom_labels += [specie] * self.aflow_data['composition'][n]
        sec_system.atoms.labels = atom_labels
        if self.aflow_data.get('positions_cartesian') is not None:
            sec_system.atoms.positions = (
                self.aflow_data.get('positions_cartesian') * ureg.angstrom
            )

        # parse system metadata from aflow_data
        system_quantities = [
            'compound',
            'prototype',
            'nspecies',
            'natoms',
            'natoms_orig',
            'composition',
            'density',
            'density_orig',
            'scintillation_attenuation_length',
            'stoichiometry',
            'species',
            'geometry',
            'geometry_orig',
            'volume_cell',
            'volume_atom',
            'volume_cell_orig',
            'volume_atom_orig',
            'n_sg',
            'sg',
            'sg2',
            'spacegroup_orig',
            'spacegroup_relax',
            'Bravais_lattice_orig',
            'lattice_variation_orig',
            'lattice_system_orig',
            'Pearson_symbol_orig',
            'Bravais_lattice_relax',
            'lattice_variation_relax',
            'lattice_system_relax',
            'Pearson_symbol_relax',
            'crystal_family_orig',
            'crystal_system_orig',
            'crystal_class_orig',
            'point_group_Hermann_Mauguin_orig',
            'point_group_Schoenflies_orig',
            'point_group_orbifold_orig',
            'point_group_type_orig',
            'point_group_order_orig',
            'point_group_structure_orig',
            'Bravais_lattice_lattice_type_orig',
            'Bravais_lattice_lattice_variation_type_orig',
            'Bravais_lattice_lattice_system_orig',
            'Bravais_superlattice_lattice_type_orig',
            'Bravais_superlattice_lattice_variation_type_orig',
            'Bravais_superlattice_lattice_system_orig',
            'Pearson_symbol_superlattice_orig',
            'reciprocal_geometry_orig',
            'reciprocal_volume_cell_orig',
            'reciprocal_lattice_type_orig',
            'reciprocal_lattice_variation_type_orig',
            'Wyckoff_letters_orig',
            'Wyckoff_multiplicities_orig',
            'Wyckoff_site_symmetries_orig',
            'crystal_family',
            'crystal_system',
            'crystal_class',
            'point_group_Hermann_Mauguin',
            'point_group_Schoenflies',
            'point_group_orbifold',
            'point_group_type',
            'point_group_order',
            'point_group_structure',
            'Bravais_lattice_lattice_type',
            'Bravais_lattice_lattice_variation_type',
            'Bravais_lattice_lattice_system',
            'Bravais_superlattice_lattice_type',
            'Bravais_superlattice_lattice_variation_type',
            'Bravais_superlattice_lattice_system',
            'Pearson_symbol_superlattice',
            'reciprocal_geometry',
            'reciprocal_volume_cell',
            'reciprocal_lattice_type',
            'reciprocal_lattice_variation_type',
            'Wyckoff_letters',
            'Wyckoff_multiplicities',
            'Wyckoff_site_symmetries',
            'prototype_label_orig',
            'prototype_params_list_orig',
            'prototype_params_values_orig',
            'prototype_label_relax',
            'prototype_params_list_relax',
            'prototype_params_values_relax',
        ]
        for key in system_quantities:
            val = self.aflow_data.get(key)
            if val is not None:
                sec_system.m_set(
                    sec_system.m_get_quantity_definition(f'x_aflow_{key}'), val
                )

        # parse method metadata from self.aflow_data
        method_quantities = [
            'code',
            'species_pp',
            'n_dft_type',
            'dft_type',
            'species_pp_version',
            'species_pp_ZVAL',
            'species_pp_AUID',
            'ldau_type',
            'ldau_l',
            'ldau_u',
            'ldau_j',
            'valence_cell_iupac',
            'valence_cell_std',
            'energy_cutoff',
            'delta_electronic_energy_convergence',
            'delta_electronic_energy_threshold',
            'kpoints_relax',
            'kpoints_static',
            'n_kpoints_bands_path',
            'kpoints_bands_path',
            'kpoints_bands_nkpts',
        ]
        sec_method = Method()
        sec_run.method.append(sec_method)
        for key in method_quantities:
            val = self.aflow_data.get(key)
            if val is not None:
                sec_method.m_set(
                    sec_method.m_get_quantity_definition(f'x_aflow_{key}'), val
                )

        # parse basic calculation quantities from self.aflow_data
        sec_scc = Calculation()
        sec_run.calculation.append(sec_scc)
        sec_scc.energy = Energy()
        sec_scc.forces = Forces()
        sec_thermo = Thermodynamics()
        sec_scc.thermodynamics.append(sec_thermo)
        if self.aflow_data.get('energy_cell') is not None:
            sec_scc.energy.total = EnergyEntry(
                value=self.aflow_data['energy_cell'] * ureg.eV
            )
        if self.aflow_data.get('forces') is not None:
            sec_scc.forces.total = ForcesEntry(
                value=self.aflow_data['forces'] * ureg.eV / ureg.angstrom
            )
        if self.aflow_data.get('enthalpy_cell') is not None:
            sec_thermo.enthalpy = self.aflow_data['enthalpy_cell'] * ureg.eV
        if self.aflow_data.get('entropy_cell') is not None:
            sec_thermo.entropy = self.aflow_data['entropy_cell'] * ureg.eV / ureg.K
        if self.aflow_data.get('calculation_time') is not None:
            sec_scc.time_calculation = self.aflow_data['calculation_time'] * ureg.s
        calculation_quantities = [
            'stress_tensor',
            'pressure_residual',
            'Pulay_stress',
            'Egap',
            'Egap_fit',
            'Egap_type',
            'enthalpy_formation_cell',
            'entropic_temperature',
            'PV',
            'spin_cell',
            'spinD',
            'spinF',
            'calculation_memory',
            'calculation_cores',
            'nbondxx',
            'agl_thermal_conductivity_300K',
            'agl_debye',
            'agl_acoustic_debye',
            'agl_gruneisen',
            'agl_heat_capacity_Cv_300K',
            'agl_heat_capacity_Cp_300K',
            'agl_thermal_expansion_300K',
            'agl_bulk_modulus_static_300K',
            'agl_bulk_modulus_isothermal_300K',
            'agl_poisson_ratio_source',
            'agl_vibrational_free_energy_300K_cell',
            'agl_vibrational_free_energy_300K_atom',
            'agl_vibrational_entropy_300K_cell',
            'agl_vibrational_entropy_300K_atom',
            'ael_poisson_ratio',
            'ael_bulk_modulus_voigt',
            'ael_bulk_modulus_reuss',
            'ael_shear_modulus_voigt',
            'ael_shear_modulus_reuss',
            'ael_bulk_modulus_vrh',
            'ael_shear_modulus_vrh',
            'ael_elastic_anisotropy',
            'ael_youngs_modulus_vrh',
            'ael_speed_sound_transverse',
            'ael_speed_sound_longitudinal',
            'ael_speed_sound_average',
            'ael_pughs_modulus_ratio',
            'ael_debye_temperature',
            'ael_applied_pressure',
            'ael_average_external_pressure',
            'ael_stiffness_tensor',
            'ael_compliance_tensor',
            'bader_net_charges',
            'bader_atomic_volumes',
            'n_files',
            'files',
            'node_CPU_Model',
            'node_CPU_Cores',
            'node_CPU_MHz',
            'node_RAM_GB',
            'catalog',
            'aflowlib_version',
            'aflowlib_date',
        ]
        for key in calculation_quantities:
            val = self.aflow_data.get(key)
            if val is not None:
                setattr(sec_scc, 'x_aflow_%s' % key, val)

        # TODO: ARUN subdirectory workflow linking
        # AEL, AGL, and APL module runs create ARUN.* subdirectories each
        # containing their own aflow.in and vasprun.xml.static.xz. These are
        # currently parsed as independent entries. Future work:
        #   - parse_ael(): discover ARUN.AEL_* entries, build a nested Workflow linking
        #     each deformation VASP run as a task, with the stiffness tensor as output.
        #   - parse_apl(): discover ARUN.APL_* entries, link each displacement VASP run
        #     as a task feeding into the force constants / phonon dispersion output.
        #   - parse_agl(): similar ..
        #   - In each case, the nested module workflow should be referenced as a task
        #     inside the top-level aflow.in workflow2, parallel to the VASP_RUN tasks.
        #   - The [VASP_RUN] sequence (relax/static/bands) and these module workflows
        #     are orthogonal — the VASP_RUN workflow does not need to know about them.

        for module in self.aflow_data.get('loop', []):
            if module == 'ael':
                self.parse_ael()
            elif module == 'agl':
                self.parse_agl()
            elif module == 'apl':
                self.parse_apl()

        # VASP runs: workflow2 + DOS+bands combination
        self.parse_vasp_runs()
