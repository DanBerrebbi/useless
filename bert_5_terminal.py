import eval
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--fichier")
    parser.add_argument("--seuil", default=.75)
    parser.add_argument("--taille_echantillon", default=150)
    parser.add_argument("--top",default=40)
    args = parser.parse_args()

    test = eval.eval(args.fichier, int(args.taille_echantillon))

    test.calculs()

    test.performances(float(args.seuil))
    test.performances_classement(int(args.top))
    test.affichage()
