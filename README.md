# Convolutional Sparse Coding


[![PyPI - Version](https://img.shields.io/pypi/v/csc.svg)](https://pypi.org/project/csc)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/csc.svg)](https://pypi.org/project/csc)

-----

## Table of Contents

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install csc
```

## Données synthétiques

Au cours de l'implémentation des expériences différentes bases de données de signaux synthétiques données ont été générées. Les derbières versions contiennent 3200 signaux, chaque signal ayant un ``id`` unique. Les signaux sont générés par batch de 200 pour un couple de valeur (``SPARSITY``, ``SNR``). Les valeurs de ces deux paramètres évoluent dans les intervalles ci-dessous :
- ``SPARSITY`` $\in \{ 2, 3, 4, 5 \} $
- ``SNR`` $\in \{ 0\text{dB}, 5\text{dB}, 10\text{dB}, 15\text{dB} \}$

On obtient ainsi 200 signaux pour chacun des 16 couples différents soit 3200 signaux au total.

### Signaux non contraints

Fichier : ``synthetic-signals-200.json``

La base de signaux non contraints est obsolète, les dernières expériences reposent sur la base de données des signaux contraints.

### Signaux contraints

Cette base de données repose sur le principe suivant :
Si un atome est à ajouter à un signal synthétique alors :
- La position de l'atome doit être différente d'au moins ``POS_ERR_THRESHOLD`` en valeur absolue par rapport aux positions des atomes déjà présents dans le signal. 
- La corrélation de l'atome avec les atomes déjà présents dans le signal doit être inférieure à ``CORR_ERR_THRESHOLD``

Paramètres pour générer le fichier ``constrained-signals-200.json`` :
- ``POS_ERR_THRESHOLD = 10``
- ``CORR_ERR_THRESHOLD = 0.75``

## Résultats

A l'issue de l'exécution d'un algorithme différentes structures de base de données ont étét envisagées au cours de l'implémentation du package. On distingue les bases de données avec différentes structure par leur préfixe.

### Structure sparVar

La plus récente étant la structure dite ``sparVar``, pour **sparsity variation**, créee originellement pour les expériences precision-recall, elle permet de représenter le résultat de l'algorithme MMP ou OMP pour différentes valeurs de ``sparsity`` pour un même signal. 


#### Structure JSON :
````
'source' = input_filename
'date' = get_today_date_str()
'algorithm' = 'Convolutional MMP-DF'
'nbBranches' = branches
'connections' = connections
'dissimilarity' = dissimilarity
'maxSparsityLevel' = max_sparsity
'batchSize' = data['batchSize']
'snrLevels' = data['snrLevels']
'signalLength' = data['signalLength']
'sparsityLevels' = data['sparsityLevels']
'dictionary' = str(self)
'mmp' : 
    [
        {
            'id' : 0,
            'snr' : snr,
            'results' : [
                {
                    'mse' : #MSE,
                    'path' : '2'
                    'delay' : #DELAY,
                    'atoms' : [
                        {'x':x, 'b':b, 'y':y, 's':s}
                        #1
                    ]
                },
                ...
                {
                    'mse' : #MSE,
                    'path' : '2-3-...-1'
                    'delay' : #DELAY,
                    'atoms' : [
                        {'x':x, 'b':b, 'y':y, 's':s}
                        #max_sparsity
                    ]
                }
                ]
        },
        {
            'id' : 1,
            ...
        },

        ...

    ]
````

### Base de données actuelles

#### ``icassp-mmpdfX-200.json``
Structure sparVar sans l'attribut `'delay'` : on ne peut pas accéder au temps de calcul du résultat.

#### ``borelli-mmpdfX-200.json``
Structure sparVar avec l'attribut ``delay`` : pour les expériences où la connaissance du temps de calcul est requise.


###

###

###

## License

`csc` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
