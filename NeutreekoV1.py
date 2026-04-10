# Projet informatique choisit : Neutreeko (Jeu 2D)

# Regles du jeu : 2 joueurs (ou 1 joueur + intelligence artificielle), 3 pions chacun.
#                 Le but est d'aligner ses pions cote a cote en direction orthogonale ou diagonale.
#                 Un seul deplacement a la fois est autorise.
#                 Un joueur peut deplacer son pion dans la direction qu'il souhaite (diagonale ou orthogonale).
#                 Tant que le pion ne rencontre pas un obstacle (un pion ou le bord du plateau de jeu) il continue.

# Commandes : - Selection du pion a deplacer : Le joueur doit placer la souris sur le pion voulu.
#             - Deplacements dans la direction voulue :
#
#                   *Orthogonales :
#                           + a droite : touche directionnelle droite (fleche droite)
#                           + a gauche : touche directionnelle gauche (fleche gauche)
#                           + en haut : touche directionnelle haut (fleche du haut)
#                           + en bas : touche directionnelle bas (fleche du bas)
#
#                   *Diagonales :
#                           + en haut a droite : e
#                           + en haut a gauche : a
#                           + en bas a droite : c
#                           + en bas a gauche : w


# Pour lancer le jeu, il suffit d'executer le code et choisir son niveau dans le shell


# Remarques :

# Nous avons teste notre programme sur plusieurs ordinateurs ayant differentes versions de python et de matplotlib.

# Le programme fonctionne sans probleme sur : - Python version 3.7.3 avec Matplotlib version 3.0.3
#                                             - Python version ? avec Matplotlib version ?(Caroline)
#                                             - Python version ? avec Matplotlib version ?(lycee)

# Nous avons rencontre un probleme avec matplotlib.pyplot.ion qui ne fonctionne pas correctement pour : Python version 3.7.1 avec matplotlib version 3.0.2

# importations

import numpy as np
import matplotlib.pyplot as mp
from random import choice, randint
from copy import deepcopy
from time import sleep


# fonction de detection position des pions


def position(G, n): # Arguments: plateau et numero du pion
    """detecte la position du pion voulu n. Donne sa ligne L et sa colonne C"""
    for i in range(5):  # i parcourt les lignes
        for j in range(5):  # j parcourt les colonnes
            if G[i, j] == n:  # Rencontre le numero correspondant
                L = i
                C = j
    return L, C


# fonctions de deplacement


def deplacement_possible_D(L, C, G):
    """detecte si le deplacement du pion selectionne a droite est possible"""
    q = 0 # compteur du nombre de cases dans lesquelles on peut deplacer le pion
    while C+q < 4 and G[L, C+q+1] == 0: # tant qu'il est possible de se deplacer
        q += 1
    return q


def deplacement_D(L, C, G, q):
    """realise un deplacement du pion selectionne vers la droite"""
    global P
    Q = deepcopy(P)
    i = 0
    while i < q : # tant qu'on peut se deplacer
        C+=1
        G[L, C], G[L, C-1] = G[L, C-1], G[L, C]
        if not (np.array_equal(Q, P)): # si P a ete modifie, mettre a jour l'affichage case par case
            affichage(P)
            mp.pause(0.3) # ralentit le deplacement (esthetique)
        i += 1


def deplacement_possible_G(L, C, G):
    """detecte si le deplacement du pion selectionne a gauche est possible"""
    q = 0 # compteur du nombre de cases dans lesquelles on peut deplacer le pion
    while C-q > 0  and G[L, C-q-1] == 0: #tant qu'il est possible de se deplacer
        q += 1
    return q


def deplacement_G(L, C, G, q):
    """realise un deplacement du pion selectionne vers la gauche"""
    global P
    Q = deepcopy(P)
    i=0
    while i < q : # tant qu'on peut se deplacer
        C-=1
        G[L, C], G[L, C+1] = G[L, C+1], G[L, C]
        if not (np.array_equal(Q, P)): # si P a ete modifie, mettre a jour l'affichage case par case
            affichage(P)
            mp.pause(0.3) # ralentit le deplacement (esthetique)
        i+=1


def deplacement_possible_B(L, C, G):
    """detecte si le deplacement du pion selectionne vers le bas est possible"""
    q = 0 # compteur du nombre de cases dans lesquelles on peut deplacer le pion
    while L+q < 4   and G[L+q+1, C] == 0: #tant qu'il est possible de se deplacer
        q += 1
    return q


def deplacement_B(L, C, G, q):
    """realise un deplacement du pion selectionne vers le bas"""
    global P
    Q = deepcopy(P)
    i = 0
    while i < q : # tant qu'on peut se deplacer
        L+=1
        G[L, C], G[L-1, C] = G[L-1, C], G[L, C]
        if not (np.array_equal(Q, P)): # si P a ete modifie, mettre a jour l'affichage case par case
            affichage(P)
            mp.pause(0.3) # ralentit le deplacement (esthetique)
        i += 1


def deplacement_possible_H(L, C, G):
    """detecte si le deplacement du pion selectionne vers le haut est possible"""
    q = 0 # compteur du nombre de cases dans lesquelles on peut deplacer le pion
    while L-q > 0 and G[L-q-1, C] == 0: #tant qu'il est possible de se deplacer
        q += 1
    return q


def deplacement_H(L, C, G, q):
    """realise un deplacement du pion selectionne vers le haut"""
    global P
    Q = deepcopy(P)
    i = 0
    while i < q : # tant qu'on peut se deplacer
        L -= 1
        G[L, C], G[L+1, C] = G[L+1, C], G[L, C]
        if not np.array_equal(Q, P): # si P a ete modifie, mettre a jour l'affichage case par case
            affichage(P)
            mp.pause(0.3) # ralentit le deplacement (esthetique)
        i += 1


def deplacement_possible_BD(L,C,G):
    """detecte si le deplacement du pion selectionne sur la diagonale en bas a droite est possible"""
    q = 0 # compteur du nombre de cases dans lesquelles on peut deplacer le pion
    while C+q < 4 and L+q < 4 and G[L+q+1, C+q+1] == 0: #tant qu'il est possible de se deplacer
        q += 1
    return q


def deplacement_BD(L, C, G, q):
    """realise un deplacement du pion selectionne sur la diagonale en bas a droite"""
    global P
    i = 0
    Q = deepcopy(P)
    while i < q : # tant qu'on peut se deplacer
        L += 1
        C += 1
        G[L, C], G[L-1, C-1] = G[L-1, C-1], G[L, C]
        if not np.array_equal(Q, P): # si P a ete modifie, mettre a jour l'affichage case par case
            affichage(P)
            mp.pause(0.3) # ralentit le deplacement (esthetique)
        i += 1


def deplacement_possible_HD(L, C, G):
    """detecte si le deplacement du pion selectionne sur la diagonale en haut a droite est possible"""
    q = 0 # compteur du nombre de cases dans lesquelles on peut deplacer le pion
    while C+q < 4 and L-q > 0 and G[L-q-1, C+q+1] == 0: #tant qu'il est possible de se deplacer
        q += 1
    return q


def deplacement_HD(L, C, G, q):
    """realise un deplacement du pion selectionne sur la diagonale en haut a droite"""
    global P
    Q = deepcopy(P)
    i = 0
    while i < q : # tant qu'on peut se deplacer
        L -= 1
        C += 1
        G[L, C], G[L+1, C-1] = G[L+1, C-1], G[L, C]
        if not (np.array_equal(Q, P)): # si P a ete modifie, mettre a jour l'affichage case par case
            affichage(P)
            mp.pause(0.3) # ralentit le deplacement (esthetique)
        i += 1


def deplacement_possible_BG(L,C,G):
    """detecte si le deplacement du pion selectionne sur la diagonale en bas a gauche est possible"""
    q = 0 # compteur du nombre de cases dans lesquelles on peut deplacer le pion
    while C-q > 0 and L+q < 4 and G[L+q+1, C-q-1] == 0: #tant qu'il est possible de se deplacer
        q += 1
    return q


def deplacement_BG(L, C, G, q):
    """realise un deplacement du pion selectionne sur la diagonale en bas a gauche"""
    global P
    Q = deepcopy(P)
    i = 0
    while i < q : # tant qu'on peut se deplacer
        L += 1
        C -= 1
        G[L, C], G[L-1, C+1] = G[L-1, C+1], G[L, C]
        if not (np.array_equal(Q,P)): # si P a ete modifie, mettre a jour l'affichage case par case
            affichage(P)
            mp.pause(0.3) # ralentit le deplacement (esthetique)
        i += 1


def deplacement_possible_HG(L, C, G):
    """detecte si le deplacement du pion selectionne sur la diagonale en haut a droite est possible"""
    q = 0 # compteur du nombre de cases dans lesquelles on peut deplacer le pion
    while C-q > 0 and L-q > 0 and G[L-q-1, C-q-1] == 0: #tant qu'il est possible de se deplacer
        q += 1
    return q


def deplacement_HG(L, C, G, q):
    """realise un deplacement du pion selectionne sur la diagonale en haut a gauche"""
    global P
    i = 0
    Q = deepcopy(P)
    while i < q:  # tant qu'on peut se deplacer
        L -= 1
        C -= 1
        G[L, C], G[L+1, C+1] = G[L+1, C+1], G[L, C]
        if not (np.array_equal(Q,P)): # si P a ete modifie, mettre a jour l'affichage case par case
            affichage(P)
            mp.pause(0.3) # ralentit le deplacement (esthetique)
        i += 1


def deplacement_Z(L, C, G, sens):
    """verifie que le deplacement du pion donne n est possible dans le sens donne puis fait le deplacement si c'est bien le cas"""
    global compteur, finale, Z, W, possibilites
    indice = possibilites.index(sens) # prend l'indice du sens donne dans possibilites
    deplacement_possible = Z[indice] # prend la valeur de la fonction deplacement_possible associee au sens donne
    q = deplacement_possible(L, C, G)
    if q > 0:  # si le deplacement est possible
        compteur += 1  # alternance des joueurs (voir Commandes(event))
        deplacement = W[indice]
        deplacement(L, C, G, q)

        finale = 0 # voir le "if" dans Deplacer


# fonctions d'affichage


def affichage_pions(P, i, couleur, ax):
    """associe une representation graphique aux pions : des rectangles"""
    (L, C) = position(P, i+1) # i+1 car n>=1
    pion = mp.Rectangle((C, L), 1, 1, 0, facecolor = couleur, edgecolor = "k", fill = True) # Affichage d'un rectangle ((sommet bas gauche), largeur, hauteur, angle, couleur, couleur du bord, rempli)
    ax.add_patch(pion) # Ajout du pion dans P


def affichage_ts_pions(P, ax):
    """associe les couleurs aux pions"""
    Lc=['yellow', 'yellow', 'yellow', 'green', 'green', 'green']
    for i in range(6):
        affichage_pions(P, i, Lc[i], ax)


def affichage(P):
    """affichage du plateau de jeu"""
    global compteur
    ax = mp.axes()
    mp.axis('equal') # Quadrillage carre
    ax.clear() # Efface l'ancien affichage
    for i in range (6):
        X, Y=[0, 5], [i, i]
        mp.plot(X, Y, "k") # On affiche les colonnes de la grille en noir (d'ou le k)
        iX, iY = [i, i], [0, 5]
        mp.plot(iX, iY, "k") # On affiche les lignes en noir
    if compteur%2 == 0 :
        mp.text(-1.3, 5.5, 'Vert',horizontalalignment='center', verticalalignment='center', fontsize=10, backgroundcolor='white')
    else :
        mp.text(-1.4, 5.5, 'Jaune', horizontalalignment='center', verticalalignment='center', fontsize=10,backgroundcolor='white')
    mp.tick_params(axis='both', which='both',bottom=False, left=False, labelbottom=False, labelleft=False) # On enleve les graduations et les numeros correspondants
    mp.title('Alignez vos pions', fontsize=25, family='cursive', backgroundcolor='white')
    mp.text(-1.3,5.8,'A toi de jouer :', horizontalalignment='center', verticalalignment='center', fontsize=10, weight='bold')
    affichage_ts_pions(P, ax)
    mp.show()


# fonction detection de fin de partie


def gagner(G, i, j, k):
    """Detecte si un joueur a gagner. P = plateau, i, j et k sont les dossards des pions"""
    (A, Z) = position(G, i)
    (E, R) = position(G, j)
    (T, Y) = position(G, k)
    if A == E and A == T:  # 3!=6 possibilites sur l'horizontale
        if (Z == R+1 and Z == Y+2) or (Z == R-1 and Z == Y-2) or (Z == R-1 and Z == Y+1) or \
                (Z == R+1 and Z == Y-1) or (Z == Y+1 and Z == R+2) or (Z == Y-1 and Z == R-2):
            return True
    elif Z == R and Z == Y:  # 3!=6 possibilites sur la verticale
        if (A==E+1 and A==T+2) or (A==E-1 and A==T-2) or (A==E-1 and A==T+1) or (A==E+1 and A==T-1) or (A==T+1 and A==E+2) or (A==T-1 and A==E-2):
            return True
    else : # 2*3!=12 possibilites pour les diagonales
        if ((A,Z)==(E+1,R+1) and (A,Z)==(T+2,Y+2)) or ((A,Z)==(E+2,R+2) and (A,Z)==(T+1,Y+1)) or ((A,Z)==(E-1,R-1) and (A,Z)==(T-2,Y-2)) or ((A,Z)==(E-2,R-2) and (A,Z)==(T-1,Y-1)) or ((A,Z)==(E-1,R-1) and (A,Z)==(T+1,Y+1)) or ((A,Z)==(E+1,R+1) and (A,Z)==(T-1,Y-1)) or ((A,Z)==(E-1,R+1) and (A,Z)==(T-2,Y+2)) or ((A,Z)==(E+1,R-1) and (A,Z)==(T+2,Y-2)) or ((A,Z)==(E-2,R+2) and (A,Z)==(T-1,Y+1)) or ((A,Z)==(E+2,R-2) and (A,Z)==(T+1,Y-1)) or ((A,Z)==(E-1,R+1) and (A,Z)==(T+1,Y-1)) or ((A,Z)==(E+1,R-1) and (A,Z)==(T-1,Y+1)):
            return True
    return False


# Fonction d'execution des ordres


def Deplacer(G, n, sens):
    """effectue les commandes de deplacement"""
    global P, compteur, finale
    (L, C) = position(G, n)
    R = deepcopy(G)
    deplacement_Z(L, C, G, sens)
    if np.array_equal(R, P) and finale == 1 : # si aleatoire() n'a pas pu etre effectue
        finale = 0
        P = aleatoire() # on reessaye
        return P
    if gagner(P, i=1, j=2, k=3): # Verifier si jaune a gagne (voir fonction gagner)
        print('victoire')
        mp.text(2.5, 3, 'VICTOIRE',horizontalalignment='center', verticalalignment='center', fontsize=50)
        mp.text(2.5, 1.5, 'JAUNE', horizontalalignment='center', verticalalignment='center', fontsize=50)
    if gagner(P, i=4, j=5, k=6): # Verifier si vert a gagne (voir fonction gagner)
        print('victoire')
        mp.text(2.5, 3, 'VICTOIRE', horizontalalignment='center', verticalalignment='center', fontsize=50)
        mp.text(2.5, 1.5, 'VERT', horizontalalignment='center', verticalalignment='center', fontsize=50)
    return G # on retourne la matrice sinon pb avec la fonction position



# Fonctions de commandes du jeu par le joueur


def Selection(event):
    """detecte le pion que le joueur veut deplacer"""
    global P, x, y
    y = int(event.xdata) # valeurs x et y de la souris
    x = int(event.ydata)
    # n = P[x, y] # dossard du pion associe
    #return n


def direction(dir):
    global directions, possibilites
    indice = directions.index(dir)
    sens = possibilites[indice]
    return sens


def Commandes(event):
    """gerer le tour de jeu, ne pas retourner si pas a ce joueur de jouer"""
    global P, compteur, difficulte, x, y
    n = P[x, y]
    dir = event.key
    if compteur%2 == 0:  # c'est le tour du joueur vert
        if n == 4 or n == 5 or n == 6 :
            sens = direction(dir)
            P = Deplacer(P, n, sens)
            if gagner(P, 4, 5, 6):  # arreter la partie si on a gagne
                mp.ioff()
                mp.pause(3)
                return mp.close(fig)
            sleep(1)  # empeche un bug ou iaNiveau2 se lance trop tot
            Commandes(event)
    elif difficulte == 1 or difficulte == 2 or difficulte == 3:  # c'est le tour de l'ia
        Alternateur()
        if gagner(P, 1, 2, 3):
            mp.ioff()
            mp.pause(3)
            return mp.close(fig)
        Commandes(event)
    elif difficulte == 4:  # c'est le tour du joueur jaune
        if n == 1 or n == 2 or n == 3:
            sens = direction(dir)
            P = Deplacer(P, n, sens)
            if gagner(P, 1, 2, 3):
                mp.pause(3)
                return mp.close()


# Intelligence Artificielle

def aleatoire():
    """fait un deplacement aleatoire d'un des pions de l'ia"""
    global P, finale, possibilites
    finale = 1
    n = randint(1, 3)
    (L, C) = position(P, n)
    P = Deplacer(P, n, choice(possibilites)) # remarque : si pas possible, voir fonction Deplacer
    return P


# N'empeche pas le joueur de gagner
def verif4(alea, n):
    for i in range (1, 4) :# si le deplacement est possible et que cela met les pions du joueur cote a cote, l'ia le fait
        if n == i:  # on ne considere pas le cas ou on compare deux situations identiques
            continue
        K = deepcopy(alea)
        (L1, C1) = position(K, i)
        (L2, C2) = position(K, n)
        if (L2 == L1+1 and C2 == C1) or (L2 == L1-1 and C2 == C1) or (L2 == L1 and C2 == C1+1) or (L2 == L1 and C2 == C1-1) or (L2 == L1+1 and C2 == C1+1) or (L2 == L1-1 and C2 == C1+1) or (L2 == L1+1 and C2 == C1-1) or (L2 == L1-1 and C2 == C1-1) :
            return True
    return False

def verif3(alea, n):
    """detecte si en faisant un deplacement le joueur(vert) peut gagner."""
    global possibilites
    for sens in possibilites : # parcourt les sens de deplacement possibles
        P1 = deepcopy(alea)
        P1 = Deplacer(P1, n, sens)
        if gagner(P1, 4, 5, 6):
            return True
    return False


def verif2(alea):
    """detecte si en faisant un deplacement le joueur(vert) peut gagner(voir verif3()). Si oui, elle renvoie True (voir iaNiveau2())"""
    for n in range(4, 7) :
        if verif3(alea, n):
            return True
    return False


def verif(L, C, P, n):
    """parcourt les deplacements possibles pour un pion donne n de l'ia. Detecte si en 1 mouvement l'ia peut gagner. Si oui, elle le fait."""
    global Z, possibilites
    for deplacement_possible in Z : # parcourt les fonctions deplacement_possible
        indice = Z.index(deplacement_possible) # indice de la fonction deplacement_possible dans Z
        sens = possibilites[indice] # prend le sens associe a ce deplacement
        q = deplacement_possible(L, C, P)
        if q > 0 : # si le deplacement est possible
            P1 = deepcopy(P) # copies de P pour tester des deplacements sans modifier P
            P1 = Deplacer(P1, n, sens)
            if gagner(P1, 1, 2, 3) : # si l'ia peut gagner avec ce deplacement, le faire
                P = Deplacer(P, n, sens)
                return True
    return False


def iaNiveau1():
    """detecte si en faisant un deplacement elle peut gagner. Si oui, elle le fait. Sinon, deplacement au hasard."""

    global compteur, P
    n = 1  # compteur des dossards des pions
    while n < 4:  # parcourt les differents pions jaunes (1,2,3)
        (L, C) = position(P, n)
        if verif(L, C, P, n):  # voir verif
            return P
        if n == 3 : # toutes les possibilites ont ete testees et il n'y a pas de deplacement gagnant
            P = aleatoire()
            compteur = 0
            affichage(P)
            return P
        n+=1


def iaNiveau2():
    """fait un deplacement dans le but d'empecher le joueur de gagner. Si ce deplacement n'existe pas, faire un deplacement aleatoire(voir aleatoire())"""
    global P, limite, possibilites
    limite += 1  # compteur qui permet d'evaluer combien de fois la fonction aleatoire a ete appelee en un tour
    while limite < 400 : # si on a fait 400 tours, on estime qu'on a teste toutes les possibilites (voir choix)
        alea = deepcopy(P)
        n = randint(1, 3)
        choix = choice(possibilites)
        alea = Deplacer(alea, n, choix)
        if np.array_equal(alea, P):  # si le deplacement n'etait pas possible on rappelle la fonction
            return iaNiveau2()
        elif verif2(alea):  # si le deplacement permet au joueur de gagner par la suite, on recommence
            return iaNiveau2()
        else:  # si le deplacement est possible et que cela ne fait pas gagner le joueur, l'ai le fait
            limite = 0
            P = Deplacer(P, n, choix)
            affichage(P)
            return P
    limite = 0  # reinitialise pour qu'au prochain deplacement on puisse entrer dans le while
    P = aleatoire()  # toutes les possibilites ont ete testees, vert gagne dans tous les cas
    return P


# probleme avec les deplacements qui se font trop vite(pas carreau par carreau)
def iaNiveau3():
    """fait un deplacement dans le but d'empecher le joueur de gagner et essaye de mettre ses pions cote a cote. Si ce deplacement n'existe pas, faire un deplacement aleatoire(voir aleatoire())"""
    global P, limite, possibilites
    limite += 1  # compteur qui permet d'evaluer combien de fois la fonction aleatoire a ete appelee en un tour
    while limite < 400:  # si on a fait 400 tours, on estime qu'on a teste toutes les possibilites (voir choix)
        alea = deepcopy(P)
        n = randint(1, 3)
        choix = choice(possibilites)
        alea = Deplacer(alea, n, choix)
        if np.array_equal(alea, P):  # si le deplacement n'etait pas possible on rappelle la fonction
            return iaNiveau3()
        elif verif2(alea):  # si le deplacement permet au joueur de gagner par la suite, on recommence
            return iaNiveau3()
        else:  # si le deplacement est possible et que cela ne fait pas gagner le joueur, l'ia le fait
            if verif4(alea, n):  # choisi le deplacement ou l'ia met ses pion cote a cote s'il existe
                limite = 0
                P = Deplacer(alea, n, choix)
                return P
            else:
                return iaNiveau3()

    limite = 0
    return iaNiveau2()


def Alternateur():
    """si l'ia peut gagner en 1 mouvement cette fonction execute ce deplacement (voir fonction verif). Sinon la fonction appelle une ia selon la difficulte choisie"""
    global compteur, P, difficulte
    n = 1  # compteur des pions jaunes
    while n < 4:  # parcourt les differents pions jaunes (1,2,3)
        (L,C) = position(P, n)
        if verif(L, C, P, n): # voir verif
            return P
        if n == 3 : # toutes les possibilites ont ete testees et il n'y a pas de deplacement gagnant
            if difficulte == 1:
                P = iaNiveau1()  # a voir selon si on veut niveau 1 ou 2
                compteur = 0
                affichage(P)
                return P
            elif difficulte == 2:
                P = iaNiveau2()
                compteur = 0
                affichage(P)
                return P
            elif difficulte == 3:
                P = iaNiveau3()
                compteur = 0
                affichage(P)
                return P
        n += 1


# Fonction de lancement du jeu

def jouer(P):
    """connecte la matrice de jeu aux commandes pour la rendre interactive et lance le jeu"""
    affichage(P)
    fig.canvas.mpl_connect('button_press_event', Selection)
    fig.canvas.mpl_connect('key_press_event', Commandes)


# Selection du niveau


def ChoixNiveau(donnee):  # verifie que l'utilisateur rentre un numero
    """verifie que le joueur rentre un numero valable dans le menu"""
    while True :
        try:
            num = float(input(donnee))
            break
        except ValueError:
            pass
    return num


def AfficherMenu(options):
    """donne les choix de difficultes possibles au joueur dans un menu (console)"""
    for i in range(len(options)):
        print("{:d}. {:s}".format(i+1, options[i]))
    choix = 0
    while choix != 1 and choix != 2 and choix != 3 and choix != 4:
        choix = ChoixNiveau("Choisisez votre mode de jeu : ")
        if choix == 1:
            P = np.array([[0, 1, 0, 2, 0], [0, 0, 4, 0, 0], [0, 0, 0, 0, 0], [0, 0, 3, 0, 0], [0, 5, 0, 6, 0]])
            difficulte = 1
        elif choix == 2:
            P = np.array([[0, 1, 0, 2, 0], [0, 0, 4, 0, 0], [0, 0, 0, 0, 0], [0, 0, 3, 0, 0], [0, 5, 0, 6, 0]])
            difficulte = 2
        elif choix == 3:
            P = np.array([[0, 1, 0, 2, 0], [0, 0, 4, 0, 0], [0, 0, 0, 0, 0], [0, 0, 3, 0, 0], [0, 5, 0, 6, 0]])
            difficulte = 3
        elif choix == 4:
            P = np.array([[0, 1, 0, 2, 0], [0, 0, 4, 0, 0], [0, 0, 0, 0, 0], [0, 0, 3, 0, 0], [0, 5, 0, 6, 0]])
            difficulte = 4  # mode multijoueur : PVP
    jouer(P)
    return P, difficulte


# Principal
global limite, compteur
fig = mp.figure()
mp.ion()  # Rend interactif l'affichage
finale = 0
limite = 0
compteur = 0
options = np.array(["Facile", "Normal", "Difficile", "PVP"])
Z = (deplacement_possible_D, deplacement_possible_G, deplacement_possible_H, deplacement_possible_B,
     deplacement_possible_HD, deplacement_possible_BD, deplacement_possible_HG, deplacement_possible_BG)
W = (deplacement_D, deplacement_G, deplacement_H, deplacement_B,
     deplacement_HD, deplacement_BD, deplacement_HG, deplacement_BG)
possibilites = ("d", "g", "h", "b", "hd", "bd", "hg", "bg")
directions = ("right", "left", "down", "up", "c", "e", "w", "a")

P, difficulte = AfficherMenu(options)
#P = np.array([[0, 1, 0, 2, 0], [0, 0, 4, 0, 0], [0, 0, 0, 0, 0], [0, 0, 3, 0, 0], [0, 5, 0, 6, 0]])
#difficulte = 1
#jouer(P)