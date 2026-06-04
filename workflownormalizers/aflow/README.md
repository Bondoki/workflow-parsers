This is a NOMAD normalizer for [AFLOW](http://www.aflowlib.org/) with VASP data. It will read Aflow.in input
and VASP output files and combines band structure and density of states
into NOMAD's unified Metainfo based Archive format.

For AFLOW, the following files will be parsed:

|Input Filename| Description|
|`aflow.in` | **Mainfile:** a text file containing the aflow output|
|`vasprun.xml.static.xz`| plain text, VASP output for density of states|
|`vasprun.xml.bands.xz`| plain text, VASP output for band structure|
|`vasprun.xml.relaxXXX.xz`| plain text, VASP output for step XXX in relaxation|


