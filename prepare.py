#!/usr/bin/env python3
import numpy as np
import os
import sys
import string
import logging
import shutil
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


def generateMDP(MDPTemplateFile, outputPrefix, timeStep, numSteps, temperature, pressure, logger=None):
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
    if logger is None:
        print(f'generateMDP: Generating {outputPrefix + ".mdp"} from template {MDPTemplateFile}...')
        print(f'Timestep (dt): {timeStep}')
        print(f'Number of simulation steps (nsteps): {numSteps}')
        print(f'Temperature: {temperature}')
        print(f'Pressure: {pressure}')
    else:
        logger.info(f'generateMDP: Generating {outputPrefix + ".mdp"} from template {MDPTemplateFile}...')
        logger.info(f'Timestep (dt): {timeStep}')
        logger.info(f'Number of simulation steps (nsteps): {numSteps}')
        logger.info(f'Temperature: {temperature}')
        logger.info(f'Pressure: {pressure}')
    with open(MDPTemplateFile, 'r') as finput:
        MDP_content = string.Template(finput.read())
    MDP_content = MDP_content.safe_substitute(dt=timeStep,
                                              nsteps=numSteps,
                                              temperature=temperature,
                                              pressure=pressure)
    with open(outputPrefix + '.mdp', 'w') as foutput:
        foutput.write(MDP_content)

def generateColvars(colvarsTemplate, outputPrefix, logger=None, **kwargs):
    if logger is None:
        print(f'generateColvars: Generating {outputPrefix + ".dat"} from template {colvarsTemplate}...')
        print('Colvars parameters:')
    else:
        logger.info(f'generateColvars: Generating {outputPrefix + ".dat"} from template {colvarsTemplate}...')
        logger.info('Colvars parameters:')
    for key, val in kwargs.items():
        if logger is None:
            print(f'{key} = {val}')
        else:
            logger.info(f'{key} = {val}')
    with open(colvarsTemplate, 'r') as finput:
        content = string.Template(finput.read())
    content = content.safe_substitute(**kwargs)
    with open(outputPrefix + '.dat', 'w') as foutput:
        foutput.write(content)

def generateShellScript(shellTemplate, outputPrefix, logger=None, **kwargs):
    if logger is None:
        print(f'generateShellScript: Generating {outputPrefix + ".sh"} from template {shellTemplate}...')
    else:
        logger.info(f'generateShellScript: Generating {outputPrefix + ".sh"} from template {shellTemplate}...')
    with open(shellTemplate, 'r') as finput:
        content = string.Template(finput.read())
    content = content.safe_substitute(**kwargs)
    with open(outputPrefix + '.sh', 'w') as foutput:
        foutput.write(content)


def mearsurePolarAngles(proteinCenter, ligandCenter):
    """
    Parameters
    ----------
    proteinCenter : numpy.array
        center-of-mass of the protein

    ligandCenter : numpy.array
        center-of-mass of the ligand

    """
    vector = ligandCenter - proteinCenter
    vector /= np.linalg.norm(vector)
    return (np.degrees(np.arccos(vector[2])),
            np.degrees(np.arctan2(vector[1], vector[0])))


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
    def __init__(self, structureFile, topologyFile, ligandOnlyStructureFile, ligandOnlyTopologyFile):
        self.logger = logging.getLogger()
        self.handler = logging.StreamHandler(sys.stdout)
        self.handler.setFormatter(logging.Formatter('%(asctime)s [BFEEGromacs][%(levelname)s]:%(message)s'))
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)
        self.logger.info('Initializing BFEEGromacs...')
        self.structureFile = structureFile
        self.topologyFile = topologyFile
        self.logger.info(f'Calling MDAnalysis to load structure {self.structureFile}.')
        self.system = Universe(self.structureFile)
        dim = self.system.dimensions
        volume = dim[0] * dim[1] * dim[2]
        self.logger.info(f'The volume of the simulation box is {volume} Å^3.')
        if isclose(volume, 0.0):
            self.logger.warning(f'The volume is too small. Maybe the structure file is a PDB file without the unit cell.')
            all_atoms = self.system.select_atoms("all")
            self.system.trajectory[0].triclinic_dimensions = get_cell(all_atoms.positions)
            dim = self.system.dimensions
            self.logger.warning(f'The unit cell has been reset to {dim[0]:12.5f} {dim[1]:12.5f} {dim[2]:12.5f} .')
            newBasename = os.path.splitext(self.structureFile)[0]
            self.structureFile = newBasename + '.new.gro'
            self.saveStructure(self.structureFile)
        self.ligandOnlyStructureFile = ligandOnlyStructureFile
        self.ligandOnlyTopologyFile = ligandOnlyTopologyFile
        self.ligandOnlySystem = Universe(self.ligandOnlyStructureFile)
        dim = self.ligandOnlySystem.dimensions
        volume = dim[0] * dim[1] * dim[2]
        self.logger.info(f'The volume of the simulation box (ligand-only system) is {volume} Å^3.')
        if isclose(volume, 0.0):
            self.logger.warning(f'The volume is too small. Maybe the structure file is a PDB file without the unit cell.')
            all_atoms = self.ligandOnlySystem.select_atoms("all")
            self.ligandOnlySystem.trajectory[0].triclinic_dimensions = get_cell(all_atoms.positions)
            dim = self.ligandOnlySystem.dimensions
            self.logger.warning(f'The unit cell has been reset to {dim[0]:12.5f} {dim[1]:12.5f} {dim[2]:12.5f} .')
            newBasename = os.path.splitext(self.ligandOnlyStructureFile)[0]
            self.ligandOnlyStructureFile = newBasename + '.new.gro'
            self.saveStructure(self.ligandOnlyStructureFile)
        self.basenames = ['001_RMSD_bound',
                          '002_euler_theta',
                          '003_euler_phi',
                          '004_euler_psi',
                          '005_polar_theta',
                          '006_polar_phi',
                          '007_r',
                          '008_RMSD_unbound']
        self.logger.info('Initialization done.')

    def saveStructure(self, outputFile, selection='all'):
        self.logger.info(f'Saving a new structure file at {outputFile} with selection ({selection}).')
        selected_atoms = self.system.select_atoms(selection)
        selected_atoms.write(outputFile)

    def setProteinHeavyAtomsGroup(self, selection):
        """
        Parameters
        ----------
        selection : str
            MDAnalysis atom selection string
        """
        self.logger.info(f'Setup the atoms group of the protein by selection: {selection}')
        self.protein = self.system.select_atoms(selection)
    
    def setLigandHeavyAtomsGroup(self, selection):
        """
        Parameters
        ----------
        selection : str
            MDAnalysis atom selection string
        """
        self.logger.info(f'Setup the atoms group of the ligand by selection: {selection}')
        self.ligand = self.system.select_atoms(selection)
        self.ligandOnly = self.ligandOnlySystem.select_atoms(selection)

    def setSolventAtomsGroup(self, selection):
        """
        Parameters
        ----------
        selection : str
            MDAnalysis atom selection string
        """
        self.logger.info(f'Setup the atoms group of the solvent molecule by selection: {selection}')
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
        self.handler.setFormatter(logging.Formatter('%(asctime)s [BFEEGromacs][001][%(levelname)s]:%(message)s'))
        generate_basename = self.basenames[0]
        self.logger.info('=' * 80)
        self.logger.info(f'Generating simulation files for {generate_basename}...')
        if not os.path.exists(generate_basename):
            self.logger.info(f'Making directory {os.path.abspath(generate_basename)}...')
            os.makedirs(generate_basename)
        # generate the MDP file
        generateMDP('001.mdp.template',
                    os.path.join(generate_basename, '001_PMF'),
                    logger=self.logger,
                    timeStep=0.002,
                    numSteps=4000000,
                    temperature=300,
                    pressure=1.01325)
        # check if the ligand and protein is selected
        if not hasattr(self, 'ligand'):
            raise RuntimeError('The atoms of the ligand have not been selected.')
        if not hasattr(self, 'protein'):
            raise RuntimeError('The atoms of the protein have not been selected.')
        # measure the COM of the protein
        protein_center = measure_center(self.protein.positions)
        # convert angstrom to nanometer and format the string
        protein_center = convert(protein_center, "angstrom", "nm")
        protein_center_str = f'({protein_center[0]}, {protein_center[1]}, {protein_center[2]})'
        self.logger.info('COM of the protein: ' + protein_center_str + '.')
        # generate the index file
        self.generateGromacsIndex(os.path.join(generate_basename, 'colvars.ndx'))
        # generate the colvars configuration
        colvars_inputfile_basename = os.path.join(generate_basename, '001_colvars')
        generateColvars('001.colvars.template',
                        colvars_inputfile_basename,
                        rmsd_bin_width=0.005,
                        rmsd_lower_boundary=0.0,
                        rmsd_upper_boundary=0.5,
                        rmsd_wall_constant=0.8368,
                        ligand_selection='BFEE_Ligand',
                        protein_selection='BFEE_Protein',
                        protein_center=protein_center_str,
                        logger=self.logger)
        # generate the reference file
        self.system.select_atoms('all').write(os.path.join(generate_basename, 'reference.xyz'))
        # generate the shell script for making the tpr file
        generateShellScript('001.generate_tpr_sh.template',
                            os.path.join(generate_basename, '001_generate_tpr'),
                            logger=self.logger,
                            MDP_FILE_TEMPLATE=os.path.relpath(os.path.abspath(os.path.join(generate_basename, '001_PMF.mdp')), os.path.abspath(generate_basename)),
                            GRO_FILE_TEMPLATE=os.path.relpath(os.path.abspath(self.structureFile), os.path.abspath(generate_basename)),
                            TOP_FILE_TEMPLATE=os.path.relpath(os.path.abspath(self.topologyFile), os.path.abspath(generate_basename)),
                            COLVARS_INPUT_TEMPLATE=os.path.relpath(os.path.abspath(colvars_inputfile_basename + '.dat'), os.path.abspath(generate_basename)))
        if not os.path.exists(os.path.join(generate_basename, 'output')):
            os.makedirs(os.path.join(generate_basename, 'output'))
        self.logger.info(f"Generation of {generate_basename} done.")
        self.logger.info('=' * 80)
    
    def generate002(self):
        self.handler.setFormatter(logging.Formatter('%(asctime)s [BFEEGromacs][002][%(levelname)s]:%(message)s'))
        generate_basename = self.basenames[1]
        self.logger.info('=' * 80)
        self.logger.info(f'Generating simulation files for {generate_basename}...')
        if not os.path.exists(generate_basename):
            self.logger.info(f'Making directory {os.path.abspath(generate_basename)}...')
            os.makedirs(generate_basename)
        # generate the MDP file
        generateMDP('002.mdp.template',
                    os.path.join(generate_basename, '002_PMF'),
                    timeStep=0.002,
                    numSteps=4000000,
                    temperature=300,
                    pressure=1.01325,
                    logger=self.logger)
        # check if the ligand and protein is selected
        if not hasattr(self, 'ligand'):
            raise RuntimeError('The atoms of the ligand have not been selected.')
        if not hasattr(self, 'protein'):
            raise RuntimeError('The atoms of the protein have not been selected.')
        # measure the COM of the protein
        protein_center = measure_center(self.protein.positions)
        # convert angstrom to nanometer and format the string
        protein_center = convert(protein_center, "angstrom", "nm")
        protein_center_str = f'({protein_center[0]}, {protein_center[1]}, {protein_center[2]})'
        self.logger.info('COM of the protein: ' + protein_center_str + '.')
        # generate the index file
        self.generateGromacsIndex(os.path.join(generate_basename, 'colvars.ndx'))
        # generate the colvars configuration
        colvars_inputfile_basename = os.path.join(generate_basename, '002_colvars')
        generateColvars('002.colvars.template',
                        colvars_inputfile_basename,
                        logger=self.logger,
                        eulerTheta_width=0.005,
                        eulerTheta_lower_boundary=-10.0,
                        eulerTheta_upper_boundary=10.0,
                        eulerTheta_wall_constant=0.8368,
                        ligand_selection='BFEE_Ligand',
                        protein_selection='BFEE_Protein',
                        protein_center=protein_center_str)
        # generate the reference file
        self.system.select_atoms('all').write(os.path.join(generate_basename, 'reference.xyz'))
        # generate the shell script for making the tpr file
        generateShellScript('002.generate_tpr_sh.template',
                            os.path.join(generate_basename, '002_generate_tpr'),
                            logger=self.logger,
                            BASENAME_001=self.basenames[0],
                            MDP_FILE_TEMPLATE=os.path.relpath(os.path.abspath(os.path.join(generate_basename, '002_PMF.mdp')), os.path.abspath(generate_basename)),
                            GRO_FILE_TEMPLATE=os.path.relpath(os.path.abspath(self.structureFile), os.path.abspath(generate_basename)),
                            TOP_FILE_TEMPLATE=os.path.relpath(os.path.abspath(self.topologyFile), os.path.abspath(generate_basename)),
                            COLVARS_INPUT_TEMPLATE=os.path.relpath(os.path.abspath(colvars_inputfile_basename + '.dat'), os.path.abspath(generate_basename)))
        # also copy the awk script to modify the colvars configuration according to the PMF minima in previous stages
        shutil.copyfile('find_min_max.awk', os.path.join(generate_basename, 'find_min_max.awk'))
        if not os.path.exists(os.path.join(generate_basename, 'output')):
            os.makedirs(os.path.join(generate_basename, 'output'))
        self.logger.info(f"Generation of {generate_basename} done.")
        self.logger.info('=' * 80)

    def generate003(self):
        self.handler.setFormatter(logging.Formatter('%(asctime)s [BFEEGromacs][003][%(levelname)s]:%(message)s'))
        generate_basename = self.basenames[2]
        self.logger.info('=' * 80)
        self.logger.info(f'Generating simulation files for {generate_basename}...')
        if not os.path.exists(generate_basename):
            self.logger.info(f'Making directory {os.path.abspath(generate_basename)}...')
            os.makedirs(generate_basename)
        # generate the MDP file
        generateMDP('003.mdp.template',
                    os.path.join(generate_basename, '003_PMF'),
                    timeStep=0.002,
                    numSteps=4000000,
                    temperature=300,
                    pressure=1.01325,
                    logger=self.logger)
        # check if the ligand and protein is selected
        if not hasattr(self, 'ligand'):
            raise RuntimeError('The atoms of the ligand have not been selected.')
        if not hasattr(self, 'protein'):
            raise RuntimeError('The atoms of the protein have not been selected.')
        # measure the COM of the protein
        protein_center = measure_center(self.protein.positions)
        # convert angstrom to nanometer and format the string
        protein_center = convert(protein_center, "angstrom", "nm")
        protein_center_str = f'({protein_center[0]}, {protein_center[1]}, {protein_center[2]})'
        self.logger.info('COM of the protein: ' + protein_center_str + '.')
        # generate the index file
        self.generateGromacsIndex(os.path.join(generate_basename, 'colvars.ndx'))
        # generate the colvars configuration
        colvars_inputfile_basename = os.path.join(generate_basename, '003_colvars')
        generateColvars('003.colvars.template',
                        colvars_inputfile_basename,
                        logger=self.logger,
                        eulerPhi_width=0.005,
                        eulerPhi_lower_boundary=-10.0,
                        eulerPhi_upper_boundary=10.0,
                        eulerPhi_wall_constant=0.8368,
                        ligand_selection='BFEE_Ligand',
                        protein_selection='BFEE_Protein',
                        protein_center=protein_center_str)
        # generate the reference file
        self.system.select_atoms('all').write(os.path.join(generate_basename, 'reference.xyz'))
        # generate the shell script for making the tpr file
        generateShellScript('003.generate_tpr_sh.template',
                            os.path.join(generate_basename, '003_generate_tpr'),
                            logger=self.logger,
                            BASENAME_001=self.basenames[0],
                            BASENAME_002=self.basenames[1],
                            MDP_FILE_TEMPLATE=os.path.relpath(os.path.abspath(os.path.join(generate_basename, '003_PMF.mdp')), os.path.abspath(generate_basename)),
                            GRO_FILE_TEMPLATE=os.path.relpath(os.path.abspath(self.structureFile), os.path.abspath(generate_basename)),
                            TOP_FILE_TEMPLATE=os.path.relpath(os.path.abspath(self.topologyFile), os.path.abspath(generate_basename)),
                            COLVARS_INPUT_TEMPLATE=os.path.relpath(os.path.abspath(colvars_inputfile_basename + '.dat'), os.path.abspath(generate_basename)))
        # also copy the awk script to modify the colvars configuration according to the PMF minima in previous stages
        shutil.copyfile('find_min_max.awk', os.path.join(generate_basename, 'find_min_max.awk'))
        if not os.path.exists(os.path.join(generate_basename, 'output')):
            os.makedirs(os.path.join(generate_basename, 'output'))
        self.logger.info(f"Generation of {generate_basename} done.")
        self.logger.info('=' * 80)

    def generate004(self):
        self.handler.setFormatter(logging.Formatter('%(asctime)s [BFEEGromacs][004][%(levelname)s]:%(message)s'))
        generate_basename = self.basenames[3]
        self.logger.info('=' * 80)
        self.logger.info(f'Generating simulation files for {generate_basename}...')
        if not os.path.exists(generate_basename):
            self.logger.info(f'Making directory {os.path.abspath(generate_basename)}...')
            os.makedirs(generate_basename)
        # generate the MDP file
        generateMDP('004.mdp.template',
                    os.path.join(generate_basename, '004_PMF'),
                    timeStep=0.002,
                    numSteps=4000000,
                    temperature=300,
                    pressure=1.01325,
                    logger=self.logger)
        # check if the ligand and protein is selected
        if not hasattr(self, 'ligand'):
            raise RuntimeError('The atoms of the ligand have not been selected.')
        if not hasattr(self, 'protein'):
            raise RuntimeError('The atoms of the protein have not been selected.')
        # measure the COM of the protein
        protein_center = measure_center(self.protein.positions)
        # convert angstrom to nanometer and format the string
        protein_center = convert(protein_center, "angstrom", "nm")
        protein_center_str = f'({protein_center[0]}, {protein_center[1]}, {protein_center[2]})'
        self.logger.info('COM of the protein: ' + protein_center_str + '.')
        # generate the index file
        self.generateGromacsIndex(os.path.join(generate_basename, 'colvars.ndx'))
        # generate the colvars configuration
        colvars_inputfile_basename = os.path.join(generate_basename, '004_colvars')
        generateColvars('004.colvars.template',
                        colvars_inputfile_basename,
                        logger=self.logger,
                        eulerPsi_width=0.005,
                        eulerPsi_lower_boundary=-10.0,
                        eulerPsi_upper_boundary=10.0,
                        eulerPsi_wall_constant=0.8368,
                        ligand_selection='BFEE_Ligand',
                        protein_selection='BFEE_Protein',
                        protein_center=protein_center_str)
        # generate the reference file
        self.system.select_atoms('all').write(os.path.join(generate_basename, 'reference.xyz'))
        # generate the shell script for making the tpr file
        generateShellScript('004.generate_tpr_sh.template',
                            os.path.join(generate_basename, '004_generate_tpr'),
                            logger=self.logger,
                            BASENAME_001=self.basenames[0],
                            BASENAME_002=self.basenames[1],
                            BASENAME_003=self.basenames[2],
                            MDP_FILE_TEMPLATE=os.path.relpath(os.path.abspath(os.path.join(generate_basename, '004_PMF.mdp')), os.path.abspath(generate_basename)),
                            GRO_FILE_TEMPLATE=os.path.relpath(os.path.abspath(self.structureFile), os.path.abspath(generate_basename)),
                            TOP_FILE_TEMPLATE=os.path.relpath(os.path.abspath(self.topologyFile), os.path.abspath(generate_basename)),
                            COLVARS_INPUT_TEMPLATE=os.path.relpath(os.path.abspath(colvars_inputfile_basename + '.dat'), os.path.abspath(generate_basename)))
        # also copy the awk script to modify the colvars configuration according to the PMF minima in previous stages
        shutil.copyfile('find_min_max.awk', os.path.join(generate_basename, 'find_min_max.awk'))
        if not os.path.exists(os.path.join(generate_basename, 'output')):
            os.makedirs(os.path.join(generate_basename, 'output'))
        self.logger.info(f"Generation of {generate_basename} done.")
        self.logger.info('=' * 80)

    def generate005(self):
        self.handler.setFormatter(logging.Formatter('%(asctime)s [BFEEGromacs][005][%(levelname)s]:%(message)s'))
        generate_basename = self.basenames[4]
        self.logger.info('=' * 80)
        self.logger.info(f'Generating simulation files for {generate_basename}...')
        if not os.path.exists(generate_basename):
            self.logger.info(f'Making directory {os.path.abspath(generate_basename)}...')
            os.makedirs(generate_basename)
        # generate the MDP file
        generateMDP('005.mdp.template',
                    os.path.join(generate_basename, '005_PMF'),
                    timeStep=0.002,
                    numSteps=4000000,
                    temperature=300,
                    pressure=1.01325,
                    logger=self.logger)
        # check if the ligand and protein is selected
        if not hasattr(self, 'ligand'):
            raise RuntimeError('The atoms of the ligand have not been selected.')
        if not hasattr(self, 'protein'):
            raise RuntimeError('The atoms of the protein have not been selected.')
        # measure the COM of the protein
        protein_center = measure_center(self.protein.positions)
        # convert angstrom to nanometer and format the string
        protein_center = convert(protein_center, "angstrom", "nm")
        protein_center_str = f'({protein_center[0]}, {protein_center[1]}, {protein_center[2]})'
        self.logger.info('COM of the protein: ' + protein_center_str + '.')
        # generate the index file
        self.generateGromacsIndex(os.path.join(generate_basename, 'colvars.ndx'))
        # generate the colvars configuration
        colvars_inputfile_basename = os.path.join(generate_basename, '005_colvars')
        # measure the current polar theta angles
        ligand_center = measure_center(self.ligand.positions)
        polar_theta, polar_phi = mearsurePolarAngles(protein_center, ligand_center)
        polar_theta_center = np.around(polar_theta, 1)
        self.logger.info(f'Measured polar angles: theta = {polar_theta:12.5f} ; phi = {polar_phi:12.5f}')
        polar_theta_width = 0.005
        polar_theta_lower = polar_theta_center - polar_theta_width * np.ceil(10 / polar_theta_width)
        polar_theta_upper = polar_theta_center + polar_theta_width * np.ceil(10 / polar_theta_width)
        generateColvars('005.colvars.template',
                        colvars_inputfile_basename,
                        logger=self.logger,
                        polarTheta_width=polar_theta_width,
                        polarTheta_lower_boundary=np.around(polar_theta_lower, 2),
                        polarTheta_upper_boundary=np.around(polar_theta_upper, 2),
                        polarTheta_wall_constant=0.8368,
                        ligand_selection='BFEE_Ligand',
                        protein_selection='BFEE_Protein',
                        protein_center=protein_center_str)
        # generate the reference file
        self.system.select_atoms('all').write(os.path.join(generate_basename, 'reference.xyz'))
        # generate the shell script for making the tpr file
        generateShellScript('005.generate_tpr_sh.template',
                            os.path.join(generate_basename, '005_generate_tpr'),
                            logger=self.logger,
                            BASENAME_001=self.basenames[0],
                            BASENAME_002=self.basenames[1],
                            BASENAME_003=self.basenames[2],
                            BASENAME_004=self.basenames[3],
                            MDP_FILE_TEMPLATE=os.path.relpath(os.path.abspath(os.path.join(generate_basename, '005_PMF.mdp')), os.path.abspath(generate_basename)),
                            GRO_FILE_TEMPLATE=os.path.relpath(os.path.abspath(self.structureFile), os.path.abspath(generate_basename)),
                            TOP_FILE_TEMPLATE=os.path.relpath(os.path.abspath(self.topologyFile), os.path.abspath(generate_basename)),
                            COLVARS_INPUT_TEMPLATE=os.path.relpath(os.path.abspath(colvars_inputfile_basename + '.dat'), os.path.abspath(generate_basename)))
        # also copy the awk script to modify the colvars configuration according to the PMF minima in previous stages
        shutil.copyfile('find_min_max.awk', os.path.join(generate_basename, 'find_min_max.awk'))
        if not os.path.exists(os.path.join(generate_basename, 'output')):
            os.makedirs(os.path.join(generate_basename, 'output'))
        self.logger.info(f"Generation of {generate_basename} done.")
        self.logger.info('=' * 80)

    def generate006(self):
        self.handler.setFormatter(logging.Formatter('%(asctime)s [BFEEGromacs][006][%(levelname)s]:%(message)s'))
        generate_basename = self.basenames[5]
        self.logger.info('=' * 80)
        self.logger.info(f'Generating simulation files for {generate_basename}...')
        if not os.path.exists(generate_basename):
            self.logger.info(f'Making directory {os.path.abspath(generate_basename)}...')
            os.makedirs(generate_basename)
        # generate the MDP file
        generateMDP('006.mdp.template',
                    os.path.join(generate_basename, '006_PMF'),
                    timeStep=0.002,
                    numSteps=4000000,
                    temperature=300,
                    pressure=1.01325,
                    logger=self.logger)
        # check if the ligand and protein is selected
        if not hasattr(self, 'ligand'):
            raise RuntimeError('The atoms of the ligand have not been selected.')
        if not hasattr(self, 'protein'):
            raise RuntimeError('The atoms of the protein have not been selected.')
        # measure the COM of the protein
        protein_center = measure_center(self.protein.positions)
        # convert angstrom to nanometer and format the string
        protein_center = convert(protein_center, "angstrom", "nm")
        protein_center_str = f'({protein_center[0]}, {protein_center[1]}, {protein_center[2]})'
        self.logger.info('COM of the protein: ' + protein_center_str + '.')
        # generate the index file
        self.generateGromacsIndex(os.path.join(generate_basename, 'colvars.ndx'))
        # generate the colvars configuration
        colvars_inputfile_basename = os.path.join(generate_basename, '006_colvars')
        # measure the current polar theta angles
        ligand_center = measure_center(self.ligand.positions)
        polar_theta, polar_phi = mearsurePolarAngles(protein_center, ligand_center)
        polar_phi_center = np.around(polar_phi, 1)
        self.logger.info(f'Measured polar angles: theta = {polar_theta:12.5f} ; phi = {polar_phi:12.5f}')
        polar_phi_width = 0.005
        polar_phi_lower = polar_phi_center - polar_phi_width * np.ceil(10 / polar_phi_width)
        polar_phi_upper = polar_phi_center + polar_phi_width * np.ceil(10 / polar_phi_width)
        generateColvars('006.colvars.template',
                        colvars_inputfile_basename,
                        logger=self.logger,
                        polarPhi_width=polar_phi_width,
                        polarPhi_lower_boundary=np.around(polar_phi_lower, 2),
                        polarPhi_upper_boundary=np.around(polar_phi_upper, 2),
                        polarPhi_wall_constant=0.8368,
                        ligand_selection='BFEE_Ligand',
                        protein_selection='BFEE_Protein',
                        protein_center=protein_center_str)
        # generate the reference file
        self.system.select_atoms('all').write(os.path.join(generate_basename, 'reference.xyz'))
        # generate the shell script for making the tpr file
        generateShellScript('006.generate_tpr_sh.template',
                            os.path.join(generate_basename, '006_generate_tpr'),
                            logger=self.logger,
                            BASENAME_001=self.basenames[0],
                            BASENAME_002=self.basenames[1],
                            BASENAME_003=self.basenames[2],
                            BASENAME_004=self.basenames[3],
                            BASENAME_005=self.basenames[4],
                            MDP_FILE_TEMPLATE=os.path.relpath(os.path.abspath(os.path.join(generate_basename, '006_PMF.mdp')), os.path.abspath(generate_basename)),
                            GRO_FILE_TEMPLATE=os.path.relpath(os.path.abspath(self.structureFile), os.path.abspath(generate_basename)),
                            TOP_FILE_TEMPLATE=os.path.relpath(os.path.abspath(self.topologyFile), os.path.abspath(generate_basename)),
                            COLVARS_INPUT_TEMPLATE=os.path.relpath(os.path.abspath(colvars_inputfile_basename + '.dat'), os.path.abspath(generate_basename)))
        # also copy the awk script to modify the colvars configuration according to the PMF minima in previous stages
        shutil.copyfile('find_min_max.awk', os.path.join(generate_basename, 'find_min_max.awk'))
        if not os.path.exists(os.path.join(generate_basename, 'output')):
            os.makedirs(os.path.join(generate_basename, 'output'))
        self.logger.info(f"Generation of {generate_basename} done.")
        self.logger.info('=' * 80)

    def generate007(self):
        self.handler.setFormatter(logging.Formatter('%(asctime)s [BFEEGromacs][007][%(levelname)s]:%(message)s'))
        generate_basename = self.basenames[6]
        self.logger.info('=' * 80)
        self.logger.info(f'Generating simulation files for {generate_basename}...')
        if not os.path.exists(generate_basename):
            self.logger.info(f'Making directory {os.path.abspath(generate_basename)}...')
            os.makedirs(generate_basename)
        # generate the MDP file
        generateMDP('007_min.mdp.template',
                    os.path.join(generate_basename, '007_Minimize'),
                    timeStep=0.002,
                    numSteps=100000,
                    temperature=300,
                    pressure=1.01325,
                    logger=self.logger)
        generateMDP('007.mdp.template',
                    os.path.join(generate_basename, '007_PMF'),
                    timeStep=0.002,
                    numSteps=80000000,
                    temperature=300,
                    pressure=1.01325,
                    logger=self.logger)
        # check if the ligand, protein and solvent is selected
        if not hasattr(self, 'ligand'):
            raise RuntimeError('The atoms of the ligand have not been selected.')
        if not hasattr(self, 'protein'):
            raise RuntimeError('The atoms of the protein have not been selected.')
        if not hasattr(self, 'solvent'):
            raise RuntimeError('The atoms of the solvent have not been selected.')
        # measure the COM of the protein
        protein_center = measure_center(self.protein.positions)
        # convert angstrom to nanometer and format the string
        protein_center = convert(protein_center, "angstrom", "nm")
        protein_center_str = f'({protein_center[0]}, {protein_center[1]}, {protein_center[2]})'
        self.logger.info('COM of the protein: ' + protein_center_str + '.')
        # generate the index file
        self.generateGromacsIndex(os.path.join(generate_basename, 'colvars.ndx'))
        # generate the colvars configuration
        colvars_inputfile_basename = os.path.join(generate_basename, '007_colvars')
        # measure the current COM distance from the ligand to protein
        ligand_center = measure_center(self.ligand.positions)
        r_center = np.sqrt(np.dot(ligand_center - protein_center, ligand_center - protein_center))
        # convert r_center to nm
        r_center = np.around(convert(r_center, 'angstrom', 'nm'), 2)
        r_width = 0.01
        # r_lower_boundary = r_center - r_lower_shift
        # r_lower_shift is default to 0.2 nm
        r_lower_shift = 0.2
        r_lower_boundary = r_center - r_lower_shift
        # r_upper_boundary = r_center + r_upper_shift
        # r_upper_shift is default to 2.1 nm
        # also we will need r_upper_shift to enlarge the solvent box
        r_upper_shift = 2.1
        r_upper_boundary = r_center + r_upper_shift
        generateColvars('007.colvars.template',
                        colvars_inputfile_basename,
                        logger=self.logger,
                        r_width=r_width,
                        r_lower_boundary=r_lower_boundary,
                        r_upper_boundary=r_upper_boundary,
                        r_wall_constant=0.8368,
                        ligand_selection='BFEE_Ligand',
                        protein_selection='BFEE_Protein',
                        protein_center=protein_center_str)
        # generate the reference file
        self.system.select_atoms('all').write(os.path.join(generate_basename, 'reference.xyz'))
        # write the solvent molecules
        self.solvent.write(os.path.join(generate_basename, 'solvent.gro'))
        # generate the shell script for making the tpr file
        new_box_x = np.around(convert(self.system.dimensions[0], 'angstrom', 'nm'), 2) + r_upper_shift
        new_box_y = np.around(convert(self.system.dimensions[1], 'angstrom', 'nm'), 2) + r_upper_shift
        new_box_z = np.around(convert(self.system.dimensions[2], 'angstrom', 'nm'), 2) + r_upper_shift
        generateShellScript('007.generate_tpr_sh.template',
                            os.path.join(generate_basename, '007_generate_tpr'),
                            logger=self.logger,
                            BASENAME_001=self.basenames[0],
                            BASENAME_002=self.basenames[1],
                            BASENAME_003=self.basenames[2],
                            BASENAME_004=self.basenames[3],
                            BASENAME_005=self.basenames[4],
                            BASENAME_006=self.basenames[5],
                            BOX_MODIFIED_GRO_TEMPLATE=os.path.relpath(os.path.abspath(os.path.join(generate_basename, 'box_modified.gro')), os.path.abspath(generate_basename)),
                            MODIFIED_TOP_TEMPLATE=os.path.relpath(os.path.abspath(os.path.join(generate_basename, 'solvated.top')), os.path.abspath(generate_basename)),
                            MODIFIED_GRO_TEMPLATE=os.path.relpath(os.path.abspath(os.path.join(generate_basename, 'solvated.gro')), os.path.abspath(generate_basename)),
                            NEW_BOX_X_TEMPLATE=f'{new_box_x:.5f}',
                            NEW_BOX_Y_TEMPLATE=f'{new_box_y:.5f}',
                            NEW_BOX_Z_TEMPLATE=f'{new_box_z:.5f}',
                            MIN_MDP_FILE_TEMPLATE=os.path.relpath(os.path.abspath(os.path.join(generate_basename, '007_Minimize.mdp')), os.path.abspath(generate_basename)),
                            MDP_FILE_TEMPLATE=os.path.relpath(os.path.abspath(os.path.join(generate_basename, '007_PMF.mdp')), os.path.abspath(generate_basename)),
                            GRO_FILE_TEMPLATE=os.path.relpath(os.path.abspath(self.structureFile), os.path.abspath(generate_basename)),
                            TOP_FILE_TEMPLATE=os.path.relpath(os.path.abspath(self.topologyFile), os.path.abspath(generate_basename)),
                            COLVARS_INPUT_TEMPLATE=os.path.relpath(os.path.abspath(colvars_inputfile_basename + '.dat'), os.path.abspath(generate_basename)))
        # also copy the awk script to modify the colvars configuration according to the PMF minima in previous stages
        shutil.copyfile('find_min_max.awk', os.path.join(generate_basename, 'find_min_max.awk'))
        if not os.path.exists(os.path.join(generate_basename, 'output')):
            os.makedirs(os.path.join(generate_basename, 'output'))
        self.logger.info(f"Generation of {generate_basename} done.")
        self.logger.info('=' * 80)

    def generate008(self):
        self.handler.setFormatter(logging.Formatter('%(asctime)s [BFEEGromacs][008][%(levelname)s]:%(message)s'))
        generate_basename = self.basenames[6]
        self.logger.info('=' * 80)
        self.logger.info(f'Generating simulation files for {generate_basename}...')
        if not os.path.exists(generate_basename):
            self.logger.info(f'Making directory {os.path.abspath(generate_basename)}...')
            os.makedirs(generate_basename)
        # generate the MDP file
        generateMDP('008.mdp.template',
                    os.path.join(generate_basename, '008_PMF'),
                    logger=self.logger,
                    timeStep=0.002,
                    numSteps=4000000,
                    temperature=300,
                    pressure=1.01325)
        # generate the index file
        if hasattr(self, 'ligandOnly'):
            self.ligandOnly.write('colvars_ligand_only.ndx', name='BFEE_Ligand_Only')
        # generate the reference file for ligand only
        # extract the positions from the host-guest binding system
        ligand_position_in_system = self.ligand.positions
        # modify positions in the ligand-only system
        self.ligandOnly.positions = ligand_position_in_system
        # write out the whole ligand-only system as reference
        self.ligandOnlySystem.select_atoms('all').write(os.path.join(generate_basename, 'reference_ligand_only.xyz'))
        # generate the shell script for making the tpr file
        generateShellScript('008.generate_tpr_sh.template',
                            os.path.join(generate_basename, '008_generate_tpr'),
                            logger=self.logger,
                            MDP_FILE_TEMPLATE=os.path.relpath(os.path.abspath(os.path.join(generate_basename, '008_PMF.mdp')), os.path.abspath(generate_basename)),
                            GRO_FILE_TEMPLATE=os.path.relpath(os.path.abspath(self.ligandOnlyStructureFile), os.path.abspath(generate_basename)),
                            TOP_FILE_TEMPLATE=os.path.relpath(os.path.abspath(self.ligandOnlyTopologyFile), os.path.abspath(generate_basename)),
                            COLVARS_INPUT_TEMPLATE=os.path.relpath(os.path.abspath(colvars_inputfile_basename + '.dat'), os.path.abspath(generate_basename)))
        if not os.path.exists(os.path.join(generate_basename, 'output')):
            os.makedirs(os.path.join(generate_basename, 'output'))
        self.logger.info(f"Generation of {generate_basename} done.")
        self.logger.info('=' * 80)

if __name__ == "__main__":
    bfee = BFEEGromacs('p41-abl.pdb', 'p41-abl.top')
    bfee.setProteinHeavyAtomsGroup('segid SH3D and not (name H*)')
    bfee.setLigandHeavyAtomsGroup('segid PPRO and not (name H*)')
    bfee.setSolventAtomsGroup('resname TIP3*')
    bfee.generate001()
    bfee.generate002()
    bfee.generate003()
    bfee.generate004()
    bfee.generate005()
    bfee.generate006()
    bfee.generate007()
    bfee.generate008()
