

FOREGROUND_GENES="nanostring_tcell_markers.txt"
SAVEDIRA="nanostring_density"
SAVEDIRB="nanostring_de"

python nanostring_compute_density.py
for obs_key in density_bw1mum density_bw10mum density_bw50mum density_bw100mum density_bw200mum density_bw500mum density_bw1000mum
do
    python nanostring_run_analysis_tcells.py --savedir $SAVEDIRA --bandwidth $obs_key --tcellsonly --foreground_genes $FOREGROUND_GENES
done
python nanostring_run_analysis_lympholicules.py --savedir $SAVEDIRB --foreground_genes $FOREGROUND_GENES
