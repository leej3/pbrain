# -*- coding: utf-8 -*-
"""Utilities."""
import numpy as np
import nibabel as nib
import pandas as pd
import sys
# sys.excepthook = lambda exctype,exc,traceback : print("{}: {}".format(exctype.__name__,exc))
def clean_csv(input_csv,output_csv):
	df = pd.read_csv(input_csv)
	df['loads'] = df[df.columns[0]].apply(check_nibload)
	(df.query('loads').
		drop('loads',inplace=False).
		to_csv(output_csv,index=False,sep=',')
		)

def check_nibload(input_path):

	try:
		nib.load(input_path).get_data()
	except Exception as e:
		print(e)
		print("Failure: ", input_path)
		return False
	return True

	

def zscore(a):
    """Return array of z-scored values."""
    a = np.asarray(a)
    return (a - a.mean()) / a.std()


def run_cmd(cmd):
    import subprocess
    pp = subprocess.run(cmd,shell=True,stdout=subprocess.PIPE,stderr= subprocess.PIPE)
    print([v.split('//')[-1] for v in pp.stderr.decode('utf-8').splitlines() ])
    return pp


def setup_exceptionhook():
    """
    Overloads default sys.excepthook with our exceptionhook handler.

    If interactive, our exceptionhook handler will invoke pdb.post_mortem;
    if not interactive, then invokes default handler.
    """
    def _pdb_excepthook(type, value, tb):
        if sys.stdin.isatty() and sys.stdout.isatty() and sys.stderr.isatty():
            import traceback
            import pdb
            traceback.print_exception(type, value, tb)
            # print()
            pdb.post_mortem(tb)
        else:
            print(
              "We cannot setup exception hook since not in interactive mode")

    sys.excepthook = _pdb_excepthook

