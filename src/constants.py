import seaborn as sns 


palette = sns.color_palette(n_colors=10)

ColorsDict = {
    'propM':palette[0], 
    'propMS':palette[1], 
    'uniform':palette[2], 
    'propM+CV':palette[3], 
    'propM+logical':palette[4], 
    'propMS+logical':palette[5], 
    'uniform+logical':palette[6], 
    'oracle':palette[7]
}

# only use dashed lines for the '+logical' methods 
LineStyleDict = {
    'propM':'-', 
    'propM+logical':'--', 
    'propMS':'-', 
    'propMS+logical':'--', 
    'uniform':'-', 
    'uniform+logical':'--', 
    'propM+CV':'-', 
    'oracle':'-'
}