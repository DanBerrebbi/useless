import eval

fichier = "C:\DALET\dataset-sts\data\sts\sick2014\SICK_trial.txt"

test=eval.eval(fichier,150)

test.calculs()


test.performances(.75)
test.performances_classement(5)
test.affichage()

import scipy

x = [a[1] for a in test.results]
y = [a[2] for a in test.results]
scipy.stats.pearsonr(x, y)