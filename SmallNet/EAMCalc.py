import os
import sys
import numpy as np
from lammps import lammps

cmds = ["-screen", "log.screen"]
lmp = lammps(cmdargs=cmds)
#========================================================
def eamenergies(geofile):
#    elementstr = ' '.join( elmkey[config] )
    lmp.command("clear")
    lmp.command("log log.equil")
    lmp.command("dimension 3")
    lmp.command("box tilt large")
    lmp.command("units metal")
    lmp.command("atom_style atomic")
    lmp.command("atom_modify map array")
    lmp.command("boundary p p p")
    lmp.command("read_data %s"%(geofile))
    lmp.command("pair_style eam")
    lmp.command("neighbor 12.0 bin")

#    lmp.command("pair_coeff * * Al.eam Al")
    lmp.command("pair_coeff * * ../Al.eam")
            
    lmp.command('variable potential equal pe/atoms')
    lmp.command('variable natoms equal atoms')
    lmp.command("run 0 pre no")
    natoms = int(lmp.extract_variable('natoms', None, 0))
    value = lmp.extract_variable('potential', None, 0)

    return value
