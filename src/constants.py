import seaborn as sns

palette = sns.color_palette(n_colors=10)
palette_2 = sns.color_palette(n_colors=14)

ColorsDict = {
    'propM': palette[0],
    'propMS': palette[1],
    'uniform': palette[2],
    'propM+CV': palette[3],
    'propM+logical': palette[4],
    'propMS+logical': palette[5],
    'uniform+logical': palette[6],
    'oracle': palette[7],
    'Bet': palette[8],
    'Hoef.': palette[9],
    'Emp. Bern.': palette_2[10],
    'Bet+logical': palette_2[11],
    'Hoef.+logical': palette_2[12],
    'Emp. Bern.+logical': palette_2[13]
}

# only use dashed lines for the '+logical' methods
LineStyleDict = {
    'propM': '-',
    'propM+logical': '--',
    'propMS': '-',
    'propMS+logical': '--',
    'uniform': '-',
    'uniform+logical': '--',
    'propM+CV': '-',
    'oracle': '-',
    'Bet': '-',
    'Hoef.': '-',
    'Emp. Bern.': '-',
    'Bet+logical': '--',
    'Hoef.+logical': '--',
    'Emp. Bern.+logical': '--',
}
