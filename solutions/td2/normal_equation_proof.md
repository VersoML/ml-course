$$J(\theta) = \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

sous forme matricielle : $$J(\theta) = (X\theta - y)^T (X\theta - y)$$

Utilisez la propriété de la tansposée matricielle : $(A-B)^T = A^T - B^T$ 

Cela donne : $$J(\theta) = (X\theta)^T (X\theta) - (X\theta)^T y - y^T (X\theta) + y^T y$$

On sait que : $(AB)^T = B^T A^T$

Donc : $$J(\theta) = \theta^T X^T X \theta - \theta^T X^T y - y^T X \theta + y^T y$$

$\theta^T X^T y$ et $y^T X \theta$ sont des scalaires (produit $1 \times m$ par $m \times 1$)

Or :$$(\theta^T X^T y)^T = y^T (X^T)^T (\theta^T)^T = y^T X \theta$$

Puisqu'ils sont égaux : $$J(\theta) = \theta^T X^T X \theta - 2\theta^T X^T y + y^T y$$


Pour trouver le $\theta$ optimal qui minimise $J(\theta)$, nous devons calculer la dérivée partielle de $J$ par rapport au vecteur $\theta$ (le gradient) et l'égaler à zéro.

$$\nabla_\theta J(\theta) = \nabla_\theta (\theta^T X^T X \theta - 2\theta^T X^T y + y^T y)$$

Rappelons deux identités de calcul matriciel (où $A$ est une matrice symétrique et $b$ un vecteur constant) :
1.  $\nabla_x (b^T x) = b$ (et pareillement $\nabla_x (x^T b) = b$)
2.  $\nabla_x (x^T A x) = 2Ax$

Appliquons ces règles à notre équation (sachant que $X^T X$ est une matrice symétrique et $y^T y$ est une constante par rapport à $\theta$) :

1.  Dérivée de $\theta^T (X^T X) \theta$ $\rightarrow$ $2 X^T X \theta$
2.  Dérivée de $-2 (X^T y)^T \theta$ $\rightarrow$ $-2 X^T y$
3.  Dérivée de $y^T y$ $\rightarrow$ $0$

Le gradient est donc :
$$\nabla_\theta J(\theta) = 2 X^T X \theta - 2 X^T y$$


Pour minimiser le coût, nous posons le gradient égal à zéro :

$$2 X^T X \theta - 2 X^T y = 0$$

On divise par 2 :
$$X^T X \theta - X^T y = 0$$

On déplace le terme négatif :
$$X^T X \theta = X^T y$$

Enfin, pour isoler $\theta$, on multiplie à gauche par l'inverse de la matrice $(X^T X)$, notée $(X^T X)^{-1}$ :

$$\theta = (X^T X)^{-1} X^T y$$