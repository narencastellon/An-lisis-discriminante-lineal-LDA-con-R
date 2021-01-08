# An-lisis-discriminante-lineal-LDA-con-R

El análisis discriminante se utiliza para predecir la probabilidad de pertenecer a una clase (o categoría) determinada en función de una o varias variables predictoras. Funciona con variables predictoras continuas y / o categóricas.

Anteriormente, hemos estudiado la regresión logística para problemas de clasificación de dos clases, es decir, cuando la variable de resultado tiene dos valores posibles (0/1, no / sí, negativo / positivo).

En comparación con la regresión logística, el análisis discriminante es más adecuado para predecir la categoría de una observación en una situación en la que la variable de resultado contiene más de dos clases. Además, es más estable que la regresión logística para problemas de clasificación de clases múltiples.

Tenga en cuenta que tanto la regresión logística como el análisis discriminante se pueden utilizar para tareas de clasificación binaria.

Aca aprenderá las técnicas y extensiones de análisis discriminante más utilizadas. Además, proporcionaremos código R para realizar los diferentes tipos de análisis.

Existe varios métodos de análisis discriminante, que abordaremos paso a paso:

* **Análisis discriminante lineal ( LDA ):** utiliza combinaciones lineales de predictores para predecir la clase de una observación determinada. Supone que las variables predictoras (p) están distribuidas normalmente y las clases tienen varianzas idénticas (para análisis univariante, $p=1$) o matrices de covarianza idénticas (para análisis multivariante, $p> 1$).

* **Análisis discriminante cuadrático ( QDA ):** más flexible que LDA. Aquí, no se supone que la matriz de covarianza de clases sea la misma.

* **Análisis discriminante de mezcla ( MDA ):** se supone que cada clase es una mezcla gaussiana de subclases.

* **Análisis discriminante flexible ( FDA ):** se utilizan combinaciones no lineales de predictores, como splines.

* **Análisis discriminante regularizado ( RDA ):** la regularización (o contracción) mejora la estimación de las matrices de covarianza en situaciones en las que el número de predictores es mayor que el número de muestras en los datos de entrenamiento. Esto conduce a una mejora del análisis discriminante.

## **1. Carga de paquetes R requeridos.**
Carga de paquetes R requeridos

`tidyverse` para una fácil visualización y manipulación de datos.
`caret` para un flujo de trabajo de aprendizaje automático sencillo.
```{r message=FALSE}
library(tidyverse)
library(caret)
theme_set(theme_classic())
```


## **2. Preparando los datos.**

Usaremos el conjunto iris de datos, para predecir especies de iris basadas en las variables predictoras Sepal.Length, Sepal.Width, Petal.Length, Petal.Width.

El análisis discriminante puede verse afectado por la escala / unidad en la que se miden las variables predictoras. Generalmente se recomienda estandarizar / normalizar el predictor continuo antes del análisis.

1. Divida los datos en entrenamiento y conjunto de prueba:

```{r}
# Cargamos la data
data("iris")
# Dividimos la data para entrenamiento en un (80%) y para la prueba en un (20%)
set.seed(123)
training.samples <- iris$Species %>%
  createDataPartition(p = 0.8, list = FALSE)
train.data <- iris[training.samples, ]
test.data <- iris[-training.samples, ]
```

2. Normaliza los datos. Las variables categóricas se ignoran automáticamente.

```{r}
# Estimar parámetros de preprocesamiento
preproc.param <- train.data %>% 
  preProcess(method = c("center", "scale"))
# Transformar los datos usando los parámetros estimados
train.transformed <- preproc.param %>% predict(train.data)
test.transformed <- preproc.param %>% predict(test.data)
```

## **3. Análisis discriminante lineal - LDA.**

El algoritmo LDA comienza por encontrar direcciones que maximizan la separación entre clases, luego usa estas direcciones para predecir la clase de individuos. Estas direcciones, llamadas discriminantes lineales, son combinaciones lineales de variables predictoras.

LDA supone que los predictores están distribuidos normalmente (distribución gaussiana) y que las diferentes clases tienen medias específicas de clase e igual varianza / covarianza.

Antes de realizar LDA, considere:

* Inspecciona las distribuciones univariadas de cada variable y asegúrese de que estén distribuidas normalmente. De lo contrario, puede transformarlos usando `log` y `root` para `distribuciones exponenciales` y `Box-Cox` para distribuciones sesgadas.
* eliminar valores atípicos de sus datos y estandarizar las variables para que su escala sea comparable.

El análisis discriminante lineal se puede calcular fácilmente utilizando la función `lda()`[paquete MASS].

**Código R de inicio rápido :**

```{r message=FALSE}
library(MASS)
# Estimación del modelo
model <- lda(Species~., data = train.transformed)
# Hacer predicciones del modelo
predictions <- model %>% predict(test.transformed)
# precisión del modelo
mean(predictions$class==test.transformed$Species)
```

**Calcular LDA :**
```{r message=FALSE}
library(MASS)
model <- lda(Species~., data = train.transformed)
model
```
LDA determina las medias del grupo y calcula, para cada individuo, la probabilidad de pertenecer a los diferentes grupos. Luego, el individuo se ve afectado por el grupo con la puntuación de probabilidad más alta.

Las salidas `lda()` contienen los siguientes elementos:

* **Probabilidades previas de grupos :** la proporción de observaciones de entrenamiento en cada grupo. Por ejemplo, hay un 31% de las observaciones de entrenamiento en el grupo setosa
* **Grupo significa :** centro de gravedad del grupo. Muestra la media de cada variable en cada grupo.
* **Coeficientes de discriminantes lineales :** muestra la combinación lineal de variables predictoras que se utilizan para formar la regla de decisión LDA.
por ejemplo:

$$LD1=0.67\times Sepal.Length+0.65\times Sepal.Width- 3.83\times Petal.Length -2.27\times Petal.Width$$,.

Del mismo modo,

$$LD2=0.04\times Sepal.Length-1\times Sepal.Width+1.44\times Petal.Length-1.96\times Petal.Width.$$

El uso de la función `plot()` produce gráficos de los discriminantes lineales, obtenidos calculando LD1 y LD2 para cada una de las observaciones de entrenamiento.

```{r}
plot(model,main="Gráfico de Discriminantes lineales")
```

**Haciendo predicciones :**
```{r}
predictions <- model %>% predict(test.transformed)
names(predictions)
```

La función `predict()` devuelve los siguientes elementos:

* **class :** clases predichas de observaciones.
* **posterior:** es una matriz cuyas columnas son los grupos, las filas son los individuos y los valores son la probabilidad posterior de que la observación correspondiente pertenezca a los grupos.
* **x :** contiene los discriminantes lineales, descritos anteriormente

Inspeccione los resultados:

```{r}
# Predicción de la clase
head(predictions$class, 6)
```
```{r}
# Probabilidades pronosticadas de pertenencia a una clase.
head(predictions$posterior, 6) 

```

```{r}
# Discriminante lineal
head(predictions$x, 10)
```

Tenga en cuenta que puede crear la gráfica LDA usando ggplot2 de la siguiente manera:
```{r}
lda.data <- cbind(train.transformed, predict(model)$x)
ggplot(lda.data, aes(LD1, LD2)) +
  geom_point(aes(color = Species))+ggtitle("Gráfico LDA")
```

**Precisión del modelo :**

Puede calcular la precisión del modelo de la siguiente manera:
```{r}
mean(predictions$class==test.transformed$Species)
```
*Se puede ver que nuestro modelo clasificó correctamente el 100% de las observaciones, lo cual es excelente.*

Tenga en cuenta que, de forma predeterminada, el límite de probabilidad utilizado para decidir la pertenencia al grupo es 0,5 (conjetura aleatoria). Por ejemplo, el número de observaciones en el grupo setosa se puede volver a calcular usando:

```{r}
sum(predictions$posterior[ ,1] >=.5)
