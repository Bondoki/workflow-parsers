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
"""Integration helpers for combining pre-existing VASP entry data into
AFLOW entries.

This module is modelled after the ``parse_vasp_runs`` machinery in the
``improved-aflow-parser-vasp-workflow`` branch, but replaces **local VASP
parsing** with **search + archive loading** because the VASP files have
already been parsed as separate entries by the electronic-parsers VASP
parser.

Design alignment with the branch
----------------------------------
* **Workflow type detection** — reads ``[VASP_RUN]`` from ``aflow.in``,
  maps it to expected roles (``relax1..N``, ``static``, ``bands``).
* **File discovery** — scans the directory for
  ``vasprun.xml.<role>.(xz|bz2)`` using the same regex.
* **Entry ID hashing** — reproduces NOMAD's internal hash
  ``base64(sha512(upload_id + mainfile))`` so string-path references
  line up with the entries created by the VASP parser.
* **DOS + bands combination** — copies ``dos_electronic`` from the
  ``static`` run and ``band_structure_electronic`` from the ``bands``
  run into a combined ``Calculation`` on the AFLOW entry, exactly as
the branch does.
* **Workflow references** — uses ``/entries/<id>/archive#<path>`` strings
  for ``TaskReference`` and ``Link`` objects, matching the branch's
  reference style.

Usage::

    # Inside AFLOWParser.parse(), after all AFLOW-specific parsing:
    from .vasp_integration import AflowVaspWorkflowBuilder
    builder = AflowVaspWorkflowBuilder(parser, archive, logger)
    builder.parse_vasp_runs()

Each step can also be called individually for debugging.
"""

import os
import re
import base64
import hashlib
import logging
from typing import Dict, List, Optional, Any

from nomad.search import search
from nomad.app.v1.models import MetadataRequired
from nomad.datamodel.metainfo.workflow import Workflow, TaskReference, Link
from runschema.calculation import Calculation, Energy, Dos, DosValues
from runschema.calculation import BandStructure, BandEnergies


# ------------------------------------------------------------------
# Helper functions (identical logic to the branch)
# ------------------------------------------------------------------

AFLOW_VASPRUN_RE = re.compile(
    r'^vasprun\.xml\.(relax(\d+)|static(\d+)?|bands)\.(xz|bz2)$'
)

VASP_RUN_ROLES = {
    'GENERATE': [],
    'STATIC': ['static'],
    'STATIC_BANDS': ['static', 'bands'],
    'RELAX': ['relax1'],
    'RELAX_STATIC': ['relax1', 'static'],
    'RELAX_STATIC_BANDS': ['relax1', 'static', 'bands'],
}


def find_vasp_runs(maindir: str) -> Dict[str, str]:
    """Scan *maindir* for ``vasprun.xml.<suffix>.xz/.bz2`` files.

    Returns an ``OrderedDict`` keyed by normalised role name, sorted in
    canonical AFLOW execution order: relax1 .. relaxN, static, bands.
    """
    found = {}
    for fname in sorted(os.listdir(maindir)):
        m = AFLOW_VASPRUN_RE.match(fname)
        if not m:
            continue
        suffix = m.group(1)
        role = re.sub(r'^static\d+$', 'static', suffix)
        if role not in found:
            found[role] = os.path.join(maindir, fname)

    def _sort_key(role):
        rm = re.match(r'^relax(\d+)$', role)
        if rm:
            return (0, int(rm.group(1)))
        return {'static': (1, 0), 'bands': (2, 0)}.get(role, (3, 0))

    return dict(sorted(found.items(), key=lambda kv: _sort_key(kv[0])))


def parse_vasp_run_directive(vasp_run_str: Optional[str]):
    """Parse the raw ``[VASP_RUN]`` value from *aflow.in*.

    Returns ``(workflow_type, n_relax)`` where *n_relax* defaults to 2
    for any RELAX variant.
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


def expected_roles_from_directive(workflow_type: str, n_relax: int) -> List[str]:
    """Build the ordered list of expected VASP run roles.

    Expands the ``'relax1'`` placeholder into
    ``['relax1', 'relax2', .., 'relaxN']``.
    """
    base = list(VASP_RUN_ROLES.get(workflow_type, []))
    if 'relax1' in base and n_relax > 1:
        relax_roles = [f'relax{i}' for i in range(1, n_relax + 1)]
        idx = base.index('relax1')
        base = base[:idx] + relax_roles + base[idx + 1 :]
    return base


def infer_workflow_type(roles: List[str]) -> str:
    """Infer the AFLOW workflow type from whichever roles were found."""
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


# ------------------------------------------------------------------
# Main builder class
# ------------------------------------------------------------------

class AflowVaspWorkflowBuilder:
    """Discovers pre-existing VASP entries and links them into the AFLOW archive.

    Args:
        parser: The ``AFLOWParser`` instance (provides ``filepath``,
                ``maindir``, ``aflow_data``).
        archive: The AFLOW ``EntryArchive`` being populated.
        logger: A logger instance (may be ``None``).
    """

    def __init__(self, parser, archive, logger=None):
        self.parser = parser
        self.archive = archive
        self.logger = logger if logger is not None else logging

    # ---- entry ID helpers (same hash as the branch) ----

    def _compute_entry_id(self, mainfile_rel: str) -> Optional[str]:
        """Reproduce NOMAD's internal hash for a mainfile path.

        Returns first 28 chars of URL-safe base64(SHA-512(upload_id + path)).
        """
        upload_id = None
        try:
            upload_id = self.archive.metadata.upload_id
        except AttributeError:
            try:
                upload_id = self.archive.m_context.upload_id
            except AttributeError:
                pass

        if upload_id is None:
            # For fully local runs with no upload context, fall back to
            # using the parent directory name as a deterministic pseudo-ID.
            upload_id = os.path.dirname(mainfile_rel)

        raw = hashlib.sha512((upload_id + mainfile_rel).encode()).digest()
        return base64.urlsafe_b64encode(raw).decode().rstrip('=')[:28]

    def _ref(self, entry_id: str, path: str) -> str:
        """Return a NOMAD archive reference string."""
        #return f'/entries/{entry_id}/archive#{path}'
        return f'../upload/archive/{entry_id}#{path}'

    # ---- section helpers ----

    @staticmethod
    def _last_system_index(child) -> int:
        try:
            return len(child.run[-1].system) - 1
        except Exception:
            return -1

    @staticmethod
    def _last_calc_index(child) -> int:
        try:
            return len(child.run[-1].calculation) - 1
        except Exception:
            return -1

    @staticmethod
    def _calculation_has_section(child, path: str) -> bool:
        """Check whether *path* resolves in *child*."""
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

    # ---- discovery via search ----

    def _discover_vasp_entries(
        self, vasp_runs: Dict[str, str], upload_prefix: str = ''
    ) -> Dict[str, Any]:
        """Find the NOMAD search entries matching each VASP file on disk.

        Returns a dict ``{role: search_result_dict}`` where each result
        contains at minimum ``entry_id`` and ``mainfile``.
        """
        discovered: Dict[str, Any] = {}

        # Resolve upload_id (metadata is preferred; fall back to context)
        upload_id = None
        try:
            upload_id = self.archive.metadata.upload_id
        except AttributeError:
            try:
                upload_id = self.archive.m_context.upload_id
            except AttributeError:
                pass

        if not upload_id:
            self.logger.warning(
                'Cannot discover VASP entries: upload_id not available.'
            )
            return discovered

        # Resolve user_id for visibility filtering (same pattern as LOBSTER)
        user_id = None
        try:
            user_id = self.archive.metadata.main_author.user_id
        except Exception:
            pass

        self.logger.info(
            'Searching for VASP entries: upload_id=%s user_id=%s',
            upload_id, user_id,
        )

        try:
            results = search(
                owner='visible',
                user_id=user_id,
                query={'upload_id': upload_id},
                required=MetadataRequired(
                    include=['entry_id', 'mainfile', 'parser_name']
                ),
            ).data
        except Exception as exc:
            self.logger.warning(
                'Search for VASP entries failed (may not be available locally).',
                exc_info=exc,
            )
            return discovered

        self.logger.info('Search returned %d total entries', len(results))

        # Index by mainfile for O(1) lookups
        by_mainfile: Dict[str, Any] = {}
        for result in results:
            parser_name = result.get('parser_name', '').lower()
            if 'vasp' not in parser_name:
                continue
            mf = result.get('mainfile')
            if mf:
                by_mainfile[mf] = result
                self.logger.debug(
                    '  VASP entry found: mainfile=%s entry_id=%s',
                    mf, result.get('entry_id'),
                )

        self.logger.info(
            'Indexed %d VASP entries by mainfile', len(by_mainfile)
        )

        for role, filepath in vasp_runs.items():
            fname = os.path.basename(filepath)
            rel_path = os.path.join(upload_prefix, fname) if upload_prefix else fname

            # Build a set of path variants to try for matching
            candidates = {rel_path, fname}
            # Normalise slashes (NOMAD search always uses /; os.path may use \)
            candidates.add(rel_path.replace('\\', '/'))
            candidates.add(
                './' + rel_path.replace('\\', '/').lstrip('./\\')
            )
            # Basename-only variants
            candidates.add(fname)
            candidates.add('./' + fname)
            # Strip leading ./ from rel_path
            candidates.add(rel_path.lstrip('./\\'))

            entry = None
            matched_path = None
            for candidate in candidates:
                if candidate in by_mainfile:
                    entry = by_mainfile[candidate]
                    matched_path = candidate
                    break

            if entry is None:
                # Diagnostic: log what we tried vs what's available
                available = sorted(by_mainfile.keys())
                self.logger.warning(
                    'No pre-existing VASP entry found for role "%s". '
                    'Tried: %s.  Available VASP mainfiles: %s',
                    role,
                    sorted(candidates),
                    available[:20],  # cap length
                )
                continue

            discovered[role] = entry
            self.logger.info(
                'Matched VASP entry %s → role "%s" (via path %s)',
                entry.get('entry_id'), role, matched_path,
            )

        return discovered

    def _load_child_archives(
        self, discovered: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Load each discovered VASP archive via ``m_context.load_archive``."""
        children: Dict[str, Any] = {}

        try:
            upload_id = self.archive.metadata.upload_id
        except AttributeError:
            self.logger.warning('Cannot load VASP archives: no upload_id')
            return children

        for role, entry in discovered.items():
            entry_id = entry.get('entry_id')
            if not entry_id:
                continue
            try:
                child = self.archive.m_context.load_archive(
                    entry_id, upload_id, None
                )
                children[role] = child
                self.logger.debug('Loaded VASP archive for role "%s"', role)
            except Exception as exc:
                self.logger.warning(
                    'Failed to load VASP archive for role "%s"', role, exc_info=exc
                )

        return children

    # ---- combination logic (identical to the branch) ----

    def _combine_dos_and_bands(
        self, roles: List[str], children: Dict[str, Any]
    ):
        """Create a combined DOS + band structure view on the AFLOW entry.

        A new ``Calculation`` is appended to ``archive.run[0]`` containing:
          - ``band_structure_electronic`` copied from the ``bands`` run
          - ``dos_electronic`` copied from the ``static`` (or last relax) run

        Neither VASP child archive is modified.
        """
        if 'bands' not in children:
            return

        bands_child = children['bands']
        bs_list = []
        efermi = None

        if bands_child.run and bands_child.run[-1].calculation:
            calc = bands_child.run[-1].calculation[-1]
            bs_list = list(calc.band_structure_electronic)
            try:
                efermi = calc.energy.fermi
            except Exception:
                pass

        if not bs_list:
            self.logger.warning(
                'No band_structure_electronic in bands run — '
                'skipping combined plot on aflow.in entry'
            )
            return

        # Prefer DOS from static; fall back to last available calculation
        dos_source = children.get('static')
        dos_list = []
        if dos_source is not None and dos_source.run and dos_source.run[-1].calculation:
            calc = dos_source.run[-1].calculation[-1]
            dos_list = list(calc.dos_electronic)
            try:
                efermi = calc.energy.fermi
            except Exception:
                pass

        if not dos_list:
            self.logger.info(
                'No electronic DOS found — band structure only will be added '
                'to aflow.in entry'
            )

        sec_combined = Calculation()
        # run[0] is the aflow.in-run created in AFLOWParser.parse()
        self.archive.run[0].calculation.append(sec_combined)

        for src_bs in bs_list:
            target_bs = BandStructure()
            sec_combined.band_structure_electronic.append(target_bs)
            for src_seg in src_bs.segment:
                target_seg = BandEnergies()
                target_bs.segment.append(target_seg)
                target_seg.kpoints = src_seg.kpoints
                target_seg.energies = src_seg.energies
                if src_seg.endpoints_labels is not None:
                    target_seg.endpoints_labels = list(src_seg.endpoints_labels)
        
        for src_dos in dos_list:
            target_dos = Dos()
            sec_combined.dos_electronic.append(target_dos)
            target_dos.energies = src_dos.energies
            for src_val in src_dos.total:
                target_dos.total.append(DosValues(value=src_val.value))

        sec_combined.energy = Energy(fermi=efermi)
        
        #sec_combined.system_ref = self._ref(dos_source.metadata.entry_id, '/run/0/system/0')
        #sec_combined.system_ref = self._ref(bands_child.metadata.entry_id, '/run/0/system/0')
        #sec_combined.method_ref = self._ref(bands_child.metadata.entry_id, '/run/0/method/0') 


        self.logger.info(
            'Created combined DOS+bands Calculation on aflow.in entry',
            n_dos=len(dos_list),
            n_bs=len(bs_list),
        )

    # ---- workflow builder (identical reference style to the branch) ----

    def _build_workflow(
        self,
        workflow_type: str,
        roles: List[str],
        vasp_runs: Dict[str, str],
        children: Dict[str, Any],
        upload_prefix: str = '',
    ):
        """Construct a ``workflow2`` on the AFLOW archive linking all runs.

        Uses string-path references (``/entries/<id>/archive#<path>``) so
        NOMAD resolves them lazily.
        """
        workflow = Workflow(name=f'AFLOW {workflow_type} Workflow')

        # Map roles → computed entry IDs (for reference strings)
        entry_ids: Dict[str, Optional[str]] = {}
        for role, filepath in vasp_runs.items():
            fname = os.path.basename(filepath)
            rel = os.path.join(upload_prefix, fname) if upload_prefix else fname
            eid = self._compute_entry_id(rel)
            entry_ids[role] = eid

        tasks: List[Any] = []
        prev_role = None

        for role in roles:
            eid = entry_ids.get(role)
            child = children.get(role)
            if eid is None or child is None:
                prev_role = role
                continue

            task = TaskReference()
            task.name = role.upper()

            # inputs: structure from previous run, or own first system
            if (
                prev_role
                and entry_ids.get(prev_role)
                and children.get(prev_role)
            ):
                prev_eid = entry_ids[prev_role]
                prev_child = children[prev_role]
                last_sys = self._last_system_index(prev_child)
                task.inputs = [
                    Link(
                        name=f'Structure from {prev_role.upper()}',
                        section=self._ref(prev_eid, f'/run/0/system/{last_sys}'),
                    )
                ]
            else:
                if self._calculation_has_section(child, '/run/0/system/0'):
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

            # task reference: prefer workflow2, fall back to calculation itself
            if self._calculation_has_section(child, '/workflow2'):
                task.task = self._ref(eid, '/workflow2')
            elif last_calc >= 0:
                task.task = self._ref(eid, f'/run/0/calculation/{last_calc}')

            tasks.append(task)
            prev_role = role

        workflow.tasks = tasks

        # Global workflow inputs / outputs
        first_role = next(
            (r for r in roles if r in entry_ids and r in children), None
        )
        last_role = next(
            (r for r in reversed(roles) if r in entry_ids and r in children), None
        )

        if first_role:
            workflow.inputs = [
                Link(
                    name='Input structure',
                    section=self._ref(entry_ids[first_role], '/run/0/system/0'),
                )
            ]
        if last_role:
            last_calc = self._last_calc_index(children[last_role])
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

    # ---- local fallback helpers ----

    def _parse_vasp_runs_locally(
        self, vasp_runs: Dict[str, str]
    ) -> Dict[str, Any]:
        """Parse VASP files directly with VASPParser when search is unavailable.

        Returns a dict ``{role: child_archive}`` just like
        ``_load_child_archives`` does.
        """
        children: Dict[str, Any] = {}

        try:
            from electronicparsers.vasp import VASPParser
            from nomad.datamodel import EntryArchive
        except ImportError as exc:
            self.logger.warning(
                'VASPParser not available — cannot parse VASP files locally. '
                'Install electronic-parsers: pip install electronic-parsers',
                exc_info=exc,
            )
            return children

        for role, filepath in vasp_runs.items():
            self.logger.debug('Locally parsing VASP run "%s"', role)
            child = EntryArchive()
            try:
                VASPParser().parse(filepath, child, self.logger)
            except Exception as exc:
                self.logger.warning(
                    'Failed to parse VASP file %s', filepath, exc_info=exc
                )
                continue
            children[role] = child
            self.logger.info('Locally parsed VASP run "%s"', role)

        return children

    def _build_direct_ref_workflow(
        self,
        workflow_type: str,
        roles: List[str],
        children: Dict[str, Any],
    ):
        """Build a workflow using direct object references (for local parsing).

        Used when NOMAD entry IDs are not available (local testing).  Creates
        ``TaskReference`` and ``Link`` objects pointing directly at the
        loaded archive sections rather than string-path references.
        """
        workflow = Workflow(name=f'AFLOW {workflow_type} Workflow')

        tasks: List[Any] = []
        for role in roles:
            child = children.get(role)
            if child is None:
                continue

            task = TaskReference()
            task.name = role.upper()

            input_structure = extract_section(child, ['run', 'system'])
            vasp_calculation = extract_section(child, ['run', 'calculation'])

            if input_structure is not None:
                task.inputs = [
                    Link(section=input_structure, name='Input Structure')
                ]
            if vasp_calculation is not None:
                task.outputs = [
                    Link(section=vasp_calculation, name='VASP Calculation')
                ]

            # Prefer workflow2 on child, fall back to last calculation
            if self._calculation_has_section(child, '/workflow2'):
                task.task = child.workflow2
            elif child.run and child.run[-1].calculation:
                task.task = child.run[-1].calculation[-1]

            tasks.append(task)

        workflow.tasks = tasks

        # Global inputs / outputs
        first_role = next((r for r in roles if r in children), None)
        last_role = next((r for r in reversed(roles) if r in children), None)

        if first_role:
            first_input = extract_section(children[first_role], ['run', 'system'])
            if first_input is not None:
                workflow.inputs = [
                    Link(section=first_input, name='Input structure')
                ]
        if last_role:
            last_calc = extract_section(children[last_role], ['run', 'calculation'])
            if last_calc is not None:
                workflow.outputs = [
                    Link(section=last_calc, name='Final result')
                ]

        self.archive.workflow2 = workflow

    # ---- top-level orchestration ----

    def parse_vasp_runs(self):
        """Main entry point — discover, load, combine, and link VASP entries.

        Called from ``AFLOWParser.parse()`` after the AFLOW-specific data
        has been written.
        """
        # 1. Detect workflow type from [VASP_RUN] directive
        vasp_run_str = None
        if hasattr(self.parser, 'aflowin_parser') and self.parser.aflowin_parser:
            vasp_run_str = getattr(self.parser.aflowin_parser, 'vasp_run', None)
        if vasp_run_str is None:
            vasp_run_str = self.parser.aflow_data.get('vasp_run')

        workflow_type, n_relax = parse_vasp_run_directive(vasp_run_str)

        # 2. Find VASP files on disk
        vasp_runs = find_vasp_runs(self.parser.maindir)
        if not vasp_runs:
            self.logger.info('No vasprun.xml.*.xz files found alongside aflow.in')
            return

        # 3. Build expected role list
        if workflow_type is not None:
            expected = expected_roles_from_directive(workflow_type, n_relax)
        else:
            self.logger.info(
                '[VASP_RUN] absent — inferring workflow type from files on disk'
            )
            expected = list(vasp_runs.keys())
            workflow_type = infer_workflow_type(expected)
            n_relax = sum(1 for r in expected if re.match(r'^relax\d+$', r))

        # Warn about missing expected roles
        for role in expected:
            if role not in vasp_runs:
                self.logger.warning(
                    'Expected role "%s" from [VASP_RUN]=%s not found on disk',
                    role, vasp_run_str,
                )

        # Merge expected + unexpected roles
        roles = list(expected)
        for role in vasp_runs:
            if role not in roles:
                self.logger.info(
                    'Unexpected VASP run "%s" found on disk — including anyway',
                    role,
                )
                roles.append(role)

        # 4. Upload prefix for entry ID hashing
        try:
            upload_root = self.archive.m_context.raw_path()
            upload_prefix = os.path.relpath(self.parser.maindir, upload_root)
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

        # 5. Discover pre-existing VASP entries via search (preferred)
        discovered = self._discover_vasp_entries(vasp_runs, upload_prefix)
        search_based = False
        children: Dict[str, Any] = {}

        if discovered:
            children = self._load_child_archives(discovered)
            if children:
                search_based = True

        if not children:
            self.logger.info(
                'No pre-existing VASP entries found via search — '
                'falling back to local VASP parsing'
            )
            children = self._parse_vasp_runs_locally(vasp_runs)

        if not children:
            self.logger.warning(
                'No VASP archives could be loaded or parsed'
            )
            return

        # 6. Combine DOS + bands into AFLOW entry
        #if 'bands' in children and len(children) > 1:
        #    self._combine_dos_and_bands(roles, children)

        # 7. Build workflow2
        if search_based:
            self._build_workflow(
                workflow_type, roles, vasp_runs, children, upload_prefix
            )
        else:
            self._build_direct_ref_workflow(workflow_type, roles, children)

        self.logger.info(
            'Built AFLOW VASP workflow with %d task(s)', len(children)
        )


# ------------------------------------------------------------------
# Stand-alone convenience function
# ------------------------------------------------------------------

def add_vasp_workflow_to_aflow(parser, archive, logger=None):
    """One-shot helper to be called from ``AFLOWParser.parse()``.

    Typical call site::

        def parse(self, filepath, archive, logger):
            # ... existing AFLOW parsing ...

            from .vasp_integration import add_vasp_workflow_to_aflow
            add_vasp_workflow_to_aflow(self, archive, logger)
    """
    builder = AflowVaspWorkflowBuilder(parser, archive, logger)
    builder.parse_vasp_runs()
