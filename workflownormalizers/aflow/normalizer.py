from typing import (
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import (
        EntryArchive,
    )
    from structlog.stdlib import (
        BoundLogger,
    )
import json
import yaml

from nomad.config import config
from nomad.normalizing import Normalizer
from nomad.utils import hash as nomad_hash

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

configuration = config.get_plugin_entry_point(
    'workflownormalizers:aflow_vasp_normalizer_entry_point'
)


class AflowVaspNormalizer(Normalizer):
    
    # normalizer_level = 3
    
    def get_reference(self, upload_id, entry_id):
        return f'../uploads/{upload_id}/archive/{entry_id}'
    
    def _ref(self, entry_id: str, path: str) -> str:
        """Return a NOMAD archive reference string."""
        #return f'/entries/{entry_id}/archive#{path}'
        return f'../upload/archive/{entry_id}#{path}'


    def get_entry_id(self, upload_id, filename):
        return nomad_hash(upload_id, filename)


    def get_hash_ref(self, upload_id, filename):
        return f'{get_reference(upload_id, get_entry_id(upload_id, filename))}#data'

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)
        logger.info('AflowVaspNormalizer.normalize', parameter=configuration.parameter)
        # if archive.results and archive.results.material:
        #     archive.results.material.elements = ['C', 'O']
                # Access parsed data from aflow.in
        
        uploadid = archive.m_context.upload_id
        #print(uploadid)
        #if dos_archive:
        self.logger.info(f'Upload id {uploadid}')
        self.logger.info(f'Raw path {archive.m_context.raw_path}')
        
        main_path = archive.metadata.mainfile.rpartition('/')[0]
        
        bands_filename = main_path+'/vasprun.xml.bands.xz' if main_path else 'vasprun.xml.bands.xz' # no directory if on root level
        #file_exists = archive.m_context.raw_path_exists(archive.metadata.mainfile.rsplit('/', 1)[0]+'/vasprun.xml.bands.xz')
        file_exists = archive.m_context.raw_path_exists(bands_filename)
        self.logger.info(archive.metadata.mainfile.rsplit('/', 1)[0]+f'/vasprun.xml.bands.xz found: {file_exists}')
        #print(file_exists)

        static_filename = main_path+'/vasprun.xml.static.xz' if main_path else 'vasprun.xml.static.xz'
        file_exists_static = archive.m_context.raw_path_exists(static_filename)
        #file_exists_static = archive.m_context.raw_path_exists(archive.metadata.mainfile.rsplit('/', 1)[0]+'/vasprun.xml.static.xz')
        self.logger.info(archive.metadata.mainfile.rsplit('/', 1)[0]+f'/vasprun.xml.static.xz found: {file_exists_static}')
        #print(file_exists)

        #print(archive.metadata.mainfile)
        self.logger.info(f'archive.metadata.mainfile: {archive.metadata.mainfile}')
        
        # ensure defaults exist
        bs_list = []
        dos_list = []
        efermi = None

        if (archive.metadata.mainfile.endswith('aflow.in')):
            
            # ensure defaults exist
            # bs_list = []
            # dos_list = []
            # efermi = None
            
            bands_archive = None
            static_archive = None
            bs_source_archive = None
            dos_source_archive = None

            if file_exists:
                main_path = archive.metadata.mainfile.rpartition('/')[0]
                bands_filename = main_path+'/vasprun.xml.bands.xz' if main_path else 'vasprun.xml.bands.xz'
                bands_archive = archive.m_context.load_archive(self.get_entry_id(uploadid, bands_filename), uploadid, None)

                if bands_archive.run and bands_archive.run[-1].calculation:
                    self.logger.info('Access bands_archive')
                    bs_list = bands_archive.run[-1].calculation[-1].band_structure_electronic
                    efermi = bands_archive.run[-1].calculation[-1].energy.fermi
                    bs_source_archive = bands_archive

            if file_exists_static:
                main_path = archive.metadata.mainfile.rpartition('/')[0]
                static_filename = main_path+'/vasprun.xml.static.xz' if main_path else 'vasprun.xml.static.xz'
                static_archive = archive.m_context.load_archive(self.get_entry_id(uploadid, static_filename), uploadid, None)

                if static_archive.run and static_archive.run[-1].calculation:
                    self.logger.info('Access static_archive')
                    dos_list = static_archive.run[-1].calculation[-1].dos_electronic
                    
                    if dos_list[-1].band_gap:
                        dos_list[-1].band_gap.clear()
                    # will overwrite bands efermi, if we can read this here.
                    # that is intended as the DOS fermi energy is more accurate
                    efermi = static_archive.run[-1].calculation[-1].energy.fermi
                    dos_source_archive = static_archive

                    # Fallback: if no bands file, take BS from static
                    if bs_source_archive is None:
                        bs_list = static_archive.run[-1].calculation[-1].band_structure_electronic
                        bs_source_archive = static_archive

            sec_combined = Calculation()
            archive.run[0].calculation.append(sec_combined)

            for bs in bs_list:
                sec_combined.band_structure_electronic.append(bs)
                self.logger.info(f'archive.metadata.mainfile: {archive.metadata.mainfile} - added bs calculation')

            for dos in dos_list:
                sec_combined.dos_electronic.append(dos)
                self.logger.info(f'archive.metadata.mainfile: {archive.metadata.mainfile} - added dos calculation')
                
            # If band structure came from the bands file, remove any band_gap
            # subsections from the DOS entries taken from static to avoid
            # duplicate band-gap information.
            if bs_source_archive is bands_archive:
                for dos in sec_combined.dos_electronic:
                    if hasattr(dos, 'band_gap') and dos.band_gap:
                        self.logger.info(
                            f'archive.metadata.mainfile: {archive.metadata.mainfile} - '
                            f'removing duplicate band_gap from DOS (using BS from bands file)'
                        )
                        dos.band_gap.clear()
                
                for bs in sec_combined.band_structure_electronic:
                    if hasattr(bs, 'band_gap') and bs.band_gap:
                        self.logger.info(
                            f'archive.metadata.mainfile: {archive.metadata.mainfile} - '
                            f'removing duplicate band_gap from BS'
                        )
                        bs.band_gap.clear()

            if bs_source_archive is not None:
                sec_combined.system_ref = self._ref(bs_source_archive.metadata.entry_id, '/run/0/system/0')
                sec_combined.method_ref = self._ref(bs_source_archive.metadata.entry_id, '/run/0/method/0')

            sec_combined.energy = Energy(fermi=efermi)


