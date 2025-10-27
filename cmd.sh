phonopy --lammps -f disp-*/forces.dump -v
phonopy --readfc -c POSCAR --dos -p 
phonopy-load --readfc --dos -p --mesh "1 1 1" --sigma 0.5 -p 
