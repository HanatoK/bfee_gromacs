#!/usr/bin/env python3
import numpy as np
import os
import string
from MDAnalysis import Universe
from MDAnalysis.units import convert
from math import isclose

def measure_minmax(atom_positions):
    """
    Mimic the VMD command "measure minmax.
    """
    xyz_array = np.transpose(atom_positions)
    min_x = np.min(xyz_array[0])
    max_x = np.max(xyz_array[0])
    min_y = np.min(xyz_array[1])
    max_y = np.max(xyz_array[1])
    min_z = np.min(xyz_array[2])
    max_z = np.max(xyz_array[2])
    return np.array([[min_x, min_y, min_z],[max_x, max_y, max_z]])


def measure_center(atom_positions):
    """
    Mimic the VMD command "measure center."
    """
    xyz_array = np.transpose(atom_positions)
    center_x = np.average(xyz_array[0])
    center_y = np.average(xyz_array[1])
    center_z = np.average(xyz_array[2])
    return np.array([center_x, center_y, center_z])


def get_cell(atom_positions):
    """
    Mimic the VMD script "get_cell.tcl".
    """
    minmax_array = measure_minmax(atom_positions)
    vec = minmax_array[1] - minmax_array[0]
    cell_basis_vector1 = np.array([vec[0], 0, 0])
    cell_basis_vector2 = np.array([0, vec[1], 0])
    cell_basis_vector3 = np.array([0, 0, vec[2]])
    return np.array([cell_basis_vector1,
                     cell_basis_vector2,
                     cell_basis_vector3])


def merge_files(filename_list, output_filename):
    """Join a list of index files into a single one"""
    with open(output_filename, "w") as foutput:
        for fn in filename_list:
            with open(fn, "r") as finput:
                for line in finput:
                    foutput.write(line)

def generateMDP(MDPTemplateFile, outputPrefix, timeStep, numSteps, temperature, pressure):
    """
    Parameters
    ----------
    MDPTemplateFile : str
        template MDP file with $dt and $nsteps

    outputPrefix : str
        prefix (no .mdp extension) of the output MDP file

    timeStep : float
        timestep for running the simulation

    numSteps : int
        number of steps for running the simulation
    """
    
    with open(MDPTemplateFile, 'r') as finput:
        MDP_content = string.Template(finput.read())
    MDP_content = MDP_content.safe_substitute(dt=timeStep,
                                              numSteps=numSteps,
                                              temperature=temperature,
                                              pressure=pressure)
    with open(outputPrefix + '.mdp', 'w') as foutput:
        foutput.write(MDP_content)

def generateColvars(colvarsTemplate, outputPrefix, **kwargs):
    with open(colvarsTemplate, 'r') as finput:
        content = string.Template(finput.read())
    content = content.safe_substitute(**kwargs)
    with open(outputPrefix + '.dat', 'w') as foutput:
        foutput.write(content)

def generateShellScript(shellTemplate, outputPrefix, **kwargs):
    with open(shellTemplate, 'r') as finput:
        content = string.Template(finput.read())
    content = content.safe_substitute(**kwargs)
    with open(outputPrefix + '.sh', 'w') as foutput:
        foutput.write(content)


class BFEEGromacs:
    """
    The entry class for handling gromacs inputs in BFEE.
    
    Attributes
    ----------
    structureFile : str
        the filename of the structfile (either in PDB or GRO format)
    topologyFile : str
        the filename of the GROMACS topology file
    
    Methods
    -------
    __init__(self, structureFile, topologyFile, indexFile=None)
        constructor of the class
    
    saveStructure(outputFile, atomSelection)
        select a group of atoms and save it

    """
    def __init__(self, structureFile, topologyFile):
        print('Initializing BFEEGromacs...')
        self.structureFile = structureFile
        self.topologyFile = topologyFile
        print(f'Calling MDAnalysis to load structure {self.structureFile}.')
        self.system = Universe(self.structureFile)
        dim = self.system.dimensions
        volume = dim[0] * dim[1] * dim[2]
        print(f'The volume of the simulation box is {volume} Å^3.')
        if isclose(volume, 0.0):
            print(f'The volume is too small. Maybe the structure file is a PDB file without the unit cell.')
            all_atoms = self.system.select_atoms("all")
            self.system.trajectory[0].triclinic_dimensions = get_cell(all_atoms.positions)
            dim = self.system.dimensions
            print(f'The unit cell has been reset to {dim[0]:12.5f} {dim[1]:12.5f} {dim[2]:12.5f} .')
        print('Initialization done.')

    def saveStructure(self, outputFile, atomSelection='all'):
        print(f'saveStructure({outputFile}) is called.')
        selected_atoms = self.system.select_atoms(atomSelection)
        selected_atoms.write(outputFile)

    def setProteinAtomsGroup(self, selection):
        """
        Parameters
        ----------
        selection : str
            MDAnalysis atom selection string
        """
        self.protein = self.system.select_atoms(selection)
    
    def setLigandAtomsGroup(self, selection):
        """
        Parameters
        ----------
        selection : str
            MDAnalysis atom selection string
        """
        self.ligand = self.system.select_atoms(selection)

    def setSolventAtomsGroup(self, selection):
        self.solvent = self.system.select_atoms(selection)

    def generateGromacsIndex(self, outputFile):
        self.system.select_atoms('all').write(outputFile, name='BFEE_all')
        if hasattr(self, 'ligand'):
            self.ligand.write(outputFile, name='BFEE_Ligand', mode='a')
        if hasattr(self, 'protein'):
            self.protein.write(outputFile, name='BFEE_Protein', mode='a')
        if hasattr(self, 'solvent'):
            self.solvent.write(outputFile, name='BFEE_Solvent', mode='a')

    def generate001(self):
        if not os.path.exists('001_RMSD_bound'):
            os.makedirs('001_RMSD_bound')
        # generate the MDP file
        generateMDP('001.mdp.template',
                    '001_RMSD_bound/001_PMF',
                    timeStep=0.002,
                    numSteps=4000000,
                    temperature=300,
                    pressure=1.01325)
        # check if the ligand and protein is selected
        if not hasattr(self, 'ligand'):
            raise RuntimeError('The atoms of the ligand has not been selected.')
        if not hasattr(self, 'protein'):
            raise RuntimeError('The atoms of the protein has not been selected.')
        # measure the COM of the protein
        protein_center = measure_center(self.protein.positions)
        # convert angstrom to nanometer and format the string
        protein_center = convert(protein_center, "angstrom", "nm")
        protein_center_str = f'({protein_center[0]}, {protein_center[1]}, {protein_center[2]})'
        # generate the index file
        self.generateGromacsIndex('001_RMSD_bound/colvars.ndx')
        # generate the colvars configuration
        generateColvars('001.colvars.template',
                        '001_RMSD_bound/001_colvars',
                        rmsd_bin_width=0.005,
                        rmsd_lower_boundary=0.0,
                        rmsd_upper_boundary=0.3,
                        rmsd_wall_constant=0.8368,
                        ligand_selection='BFEE_Ligand',
                        protein_selection='BFEE_Protein',
                        protein_center=protein_center_str)
        # generate the reference file
        self.system.select_atoms('all').write('001_RMSD_bound/reference.xyz')
        # generate the shell script for making the tpr file
        dirname = os.path.dirname(__file__)
        generateShellScript('001.generate_tpr_sh.template',
                            '001_RMSD_bound/001_generate_tpr',
                            MDP_FILE_TEMPLATE=os.path.relpath(os.path.abspath('001_RMSD_bound/001_PMF.mdp'), os.path.abspath('001_RMSD_bound/')),
                            GRO_FILE_TEMPLATE=os.path.relpath(os.path.abspath(self.structureFile), os.path.abspath('001_RMSD_bound/')),
                            TOP_FILE_TEMPLATE=os.path.relpath(os.path.abspath(self.topologyFile), os.path.abspath('001_RMSD_bound/')))
        if not os.path.exists('001_RMSD_bound/output'):
            os.makedirs('001_RMSD_bound/output')


if __name__ == "__main__":
    bfee = BFEEGromacs('p41-abl.pdb', 'p41-abl.top')
    bfee.setProteinAtomsGroup('segid SH3D and not (name H*)')
    bfee.setLigandAtomsGroup('segid PPRO and not (name H*)')
    bfee.setSolventAtomsGroup('resname TIP3*')
    bfee.generate001()
