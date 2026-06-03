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
"""Integration helpers for combining VASP entry data into AFLOW entries.

When an AFLOW mainfile (``aflow.in``) is co-located with compressed VASP
output files (``vasprun.xml.???.xz``) that are parsed as separate VASP
entries, this module discovers those entries, creates workflow links, and
optionally copies electronic band structures and density of states into
the AFLOW archive.

Usage::

    from .vasp_integration import AflowVaspIntegration

    # Inside AFLOWParser.parse(), after the initial parsing:
    integration = AflowVaspIntegration(parser, archive, logger)
    integration.add_vasp_workflow()          # link VASP entries as workflow tasks
    integration.copy_vasp_band_structures()  # optionally copy BS data
    integration.copy_vasp_dos()              # optionally copy DOS data

Design rationale
------------------
* **Discovery** is done via :py:func:`nomad.search.search` because the VASP
  entries are separate mainfile-entries in the same upload (and usually the
  same directory).
* **Archive loading** uses :py:meth:`archive.m_context.load_archive`, the
  same pattern used by the LOBSTER parser in this repository.
* **Workflow linking** uses :py:class:`TaskReference` and :py:class:`Link`
  from ``nomad.datamodel.metainfo.workflow``, which creates lazy references
  that resolve when the data is accessed.
* **Data copying** (optional) creates new metainfo objects for band
  structures / DOS rather than sharing references, avoiding side effects
  between archives.
"""

import os
from typing import List, Dict, Optional, Any

from nomad.search import search
from nomad.app.v1.models import MetadataRequired
from nomad.datamodel.metainfo.workflow import TaskReference, Link, Task
from nomad.utils import extract_section
from simulationworkflowschema import SerialSimulation
from runschema.calculation import (
    Calculation,
    Dos,
    DosValues,
    BandStructure,
    BandEnergies,
)


class AflowVaspIntegration:
    """Discovers VASP entries related to an AFLOW entry and merges their data.

    Args:
        parser: The ``AFLOWParser`` instance (provides ``filepath``, ``maindir``).
        archive: The AFLOW ``EntryArchive`` being populated.
        logger: A logger instance (may be ``None``).
    """

    def __init__(self, parser, archive, logger=None):
        self.parser = parser
        self.archive = archive
        self.logger = logger
        self._vasp_entries: List[Dict[str, Any]] = []
        self._vasp_archives: List[Any] = []

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def find_vasp_entries(self) -> List[Dict[str, Any]]:
        """Search the upload for VASP entries in the same directory as ``aflow.in``.

        Returns:
            List of search-result dicts with keys ``entry_id``, ``mainfile``,
            ``parser_name``.
        """
        self._vasp_entries = []

        try:
            upload_id = self.archive.metadata.upload_id
        except AttributeError:
            if self.logger:
                self.logger.warning(
                    'Cannot discover VASP entries: upload_id not set on archive.'
                )
            return self._vasp_entries

        parent_dir = os.path.dirname(self.parser.filepath)

        try:
            results = search(
                owner='visible',
                query={'upload_id': upload_id},
                required=MetadataRequired(
                    include=['entry_id', 'mainfile', 'parser_name']
                ),
            ).data
        except Exception as exc:
            if self.logger:
                self.logger.warning(
                    'Search for VASP entries failed.', exc_info=exc
                )
            return self._vasp_entries

        for result in results:
            if 'vasp' not in result.get('parser_name', '').lower():
                continue
            entry_mainfile = result.get('mainfile')
            if not entry_mainfile:
                continue
            # Match entries that sit in exactly the same directory.
            # In AFLOW, VASP runs for AEL/AGL/APL are stored in sub-folders
            # (e.g. ``ARUN.AEL_0_SF_N_1_0.99/vasprun.xml.001.xz``), so we
            # also accept sub-directories.
            entry_dir = os.path.dirname(entry_mainfile)
            if entry_dir == parent_dir or entry_dir.startswith(
                parent_dir + os.sep
            ):
                self._vasp_entries.append(result)

        if self.logger:
            self.logger.info(
                'Discovered %d VASP entries for AFLOW entry.',
                len(self._vasp_entries),
            )

        return self._vasp_entries

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_vasp_archives(self) -> List[Any]:
        """Load every discovered VASP archive via ``m_context.load_archive``.

        Returns:
            List of loaded VASP ``EntryArchive`` objects.
        """
        self._vasp_archives = []
        if not self._vasp_entries:
            self.find_vasp_entries()

        try:
            upload_id = self.archive.metadata.upload_id
        except AttributeError:
            if self.logger:
                self.logger.warning('Cannot load VASP archives: no upload_id.')
            return self._vasp_archives

        for entry in self._vasp_entries:
            entry_id = entry.get('entry_id')
            mainfile = entry.get('mainfile')
            if not entry_id:
                continue
            try:
                vasp_archive = self.archive.m_context.load_archive(
                    entry_id, upload_id, None
                )
                self._vasp_archives.append(vasp_archive)
                if self.logger:
                    self.logger.debug(
                        'Loaded VASP archive for %s', mainfile
                    )
            except Exception as exc:
                if self.logger:
                    self.logger.warning(
                        'Failed to load VASP archive %s.', mainfile, exc_info=exc
                    )

        return self._vasp_archives

    # ------------------------------------------------------------------
    # Workflow construction
    # ------------------------------------------------------------------

    def add_vasp_workflow(self, workflow_name: str = 'AFLOW VASP Workflow'):
        """Create a ``SerialSimulation`` workflow that links all VASP entries.

        Each VASP entry becomes a task inside the workflow.  The AFLOW
        workflow itself is stored in ``archive.workflow2``.

        The tasks are ``TaskReference`` objects pointing to each VASP entry's
        ``workflow2`` section.  This keeps the data in the original VASP
        archives and avoids duplication.
        """
        if not self._vasp_archives:
            self.load_vasp_archives()
        if not self._vasp_archives:
            return

        workflow = SerialSimulation(name=workflow_name)

        # Optionally promote an existing AFLOW sub-workflow (Elastic, Phonon …)
        # into the first task so it is not lost.
        existing_workflow = getattr(self.archive, 'workflow2', None)
        if existing_workflow is not None:
            aflow_task = TaskReference(task=existing_workflow, name='AFLOW run')
            workflow.tasks.append(aflow_task)

        for idx, vasp_archive in enumerate(self._vasp_archives, start=1):
            vasp_workflow = getattr(vasp_archive, 'workflow2', None)
            if vasp_workflow is None:
                # Fallback: craft a minimal workflow from the VASP run data
                vasp_workflow = SerialSimulation(
                    name=f'VASP calculation {idx}'
                )

            task = TaskReference(task=vasp_workflow)
            task.name = f'VASP calculation {idx}'

            # Extract a structure and calculation section from the VASP
            # archive so that the task has explicit inputs / outputs.
            input_structure = extract_section(
                vasp_archive, ['run', 'system']
            )
            vasp_calculation = extract_section(
                vasp_archive, ['run', 'calculation']
            )

            if input_structure is not None:
                task.inputs = [
                    Link(section=input_structure, name='Input Structure')
                ]
            if vasp_calculation is not None:
                task.outputs = [
                    Link(section=vasp_calculation, name='VASP Calculation')
                ]

            workflow.tasks.append(task)

        # Store the combined workflow
        self.archive.workflow2 = workflow

        if self.logger:
            self.logger.info(
                'Built AFLOW-VASP workflow with %d VASP task(s).',
                len(self._vasp_archives),
            )

    # ------------------------------------------------------------------
    # Data extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_vasp_calculation(vasp_archive) -> Optional[Any]:
        """Return the last calculation from the first ``run`` in a VASP archive."""
        try:
            run = vasp_archive.run[0]
            if not run.calculation:
                return None
            return run.calculation[-1]
        except Exception:
            return None

    @staticmethod
    def _copy_band_structure(
        source_bse: BandStructure, target_calc: Calculation
    ) -> BandStructure:
        """Deep-copy a ``band_structure_electronic`` entry into *target_calc*.

        Creates fresh metainfo objects so that the new data belongs to the
        AFLOW archive and not to the VASP archive.
        """
        target_bse = BandStructure()
        target_calc.band_structure_electronic.append(target_bse)

        for source_seg in source_bse.segment:
            target_seg = BandEnergies()
            target_bse.segment.append(target_seg)

            target_seg.kpoints = source_seg.kpoints
            target_seg.energies = source_seg.energies
            if source_seg.endpoints_labels is not None:
                target_seg.endpoints_labels = list(source_seg.endpoints_labels)

        return target_bse

    @staticmethod
    def _copy_dos(source_dos: Dos, target_calc: Calculation) -> Dos:
        """Deep-copy a ``dos_electronic`` entry into *target_calc*.

        Copies energies and the ``total`` channel.  If atom/orbital-projected
        channels exist they are copied as well.
        """
        target_dos = Dos()
        target_calc.dos_electronic.append(target_dos)

        target_dos.energies = source_dos.energies

        if source_dos.total:
            for source_val in source_dos.total:
                target_val = DosValues(value=source_val.value)
                target_dos.total.append(target_val)

        if source_dos.partial:
            for source_val in source_dos.partial:
                target_val = DosValues(
                    value=source_val.value,
                    atom_label=getattr(source_val, 'atom_label', None),
                    m_state=getattr(source_val, 'm_state', None),
                    l_state=getattr(source_val, 'l_state', None),
                )
                target_dos.partial.append(target_val)

        return target_dos

    # ------------------------------------------------------------------
    # Public copy API
    # ------------------------------------------------------------------

    def copy_vasp_band_structures(self):
        """Copy every electronic band structure from each VASP entry into AFLOW.

        A new ``Calculation`` is appended to the *last* AFLOW ``run`` for
        every VASP entry that contains a ``band_structure_electronic``.
        """  # noqa: D401
        if not self._vasp_archives:
            self.load_vasp_archives()
        if not self._vasp_archives:
            return

        try:
            aflow_run = self.archive.run[-1]
        except IndexError:
            if self.logger:
                self.logger.warning(
                    'No run section in AFLOW archive; skipping band-structure copy.'
                )
            return

        copied = 0
        for vasp_archive in self._vasp_archives:
            vasp_calc = self._get_vasp_calculation(vasp_archive)
            if vasp_calc is None or not vasp_calc.band_structure_electronic:
                continue

            aflow_calc = Calculation()
            aflow_run.calculation.append(aflow_calc)

            for source_bse in vasp_calc.band_structure_electronic:
                self._copy_band_structure(source_bse, aflow_calc)
                copied += 1

        if self.logger:
            self.logger.info(
                'Copied %d electronic band structure(s) from VASP into AFLOW.',
                copied,
            )

    def copy_vasp_dos(self):
        """Copy every electronic DOS from each VASP entry into AFLOW.

        A new ``Calculation`` is appended to the *last* AFLOW ``run`` for
        every VASP entry that contains a ``dos_electronic``.
        """  # noqa: D401
        if not self._vasp_archives:
            self.load_vasp_archives()
        if not self._vasp_archives:
            return

        try:
            aflow_run = self.archive.run[-1]
        except IndexError:
            if self.logger:
                self.logger.warning(
                    'No run section in AFLOW archive; skipping DOS copy.'
                )
            return

        copied = 0
        for vasp_archive in self._vasp_archives:
            vasp_calc = self._get_vasp_calculation(vasp_archive)
            if vasp_calc is None or not vasp_calc.dos_electronic:
                continue

            aflow_calc = Calculation()
            aflow_run.calculation.append(aflow_calc)

            for source_dos in vasp_calc.dos_electronic:
                self._copy_dos(source_dos, aflow_calc)
                copied += 1

        if self.logger:
            self.logger.info(
                'Copied %d electronic DOS object(s) from VASP into AFLOW.',
                copied,
            )

    # ------------------------------------------------------------------
    # Convenience one-shot helpers
    # ------------------------------------------------------------------

    def add_workflow_and_copy_data(self):
        """Run the full integration: workflow + band structure + DOS."""
        self.add_vasp_workflow()
        self.copy_vasp_band_structures()
        self.copy_vasp_dos()

    def add_workflow_keep_references(self):
        """Run only workflow linking; do **not** copy band structure / DOS data.

        This is the preferred mode when disk space or data duplication is a
        concern, because the AFLOW entry will only hold lightweight
        ``TaskReference`` and ``Link`` objects pointing at the VASP archives.
        """
        self.add_vasp_workflow()


# --------------------------------------------------------------------------
# Stand-alone integration function (drop-in for AFLOWParser.parse)
# --------------------------------------------------------------------------

def add_vasp_entries_to_aflow(
    parser,
    archive,
    logger=None,
    *,
    copy_bandstructure: bool = False,
    copy_dos: bool = False,
) -> AflowVaspIntegration:
    """Discover VASP entries, build workflow links, and optionally copy data.

    Typical call site inside ``AFLOWParser.parse()``::

        def parse(self, filepath, archive, logger):
            # ... existing AFLOW parsing ...

            from .vasp_integration import add_vasp_entries_to_aflow
            add_vasp_entries_to_aflow(
                self, archive, logger,
                copy_bandstructure=True, copy_dos=True
            )

    Args:
        parser: The ``AFLOWParser`` instance.
        archive: The ``EntryArchive`` being populated.
        logger: Logger (may be ``None``).
        copy_bandstructure: If ``True``, copy electronic band-structure data.
        copy_dos: If ``True``, copy electronic DOS data.

    Returns:
        The :py:class:`AflowVaspIntegration` instance that performed the work.
    """
    integration = AflowVaspIntegration(parser, archive, logger)
    integration.add_vasp_workflow()
    if copy_bandstructure:
        integration.copy_vasp_band_structures()
    if copy_dos:
        integration.copy_vasp_dos()
    return integration
