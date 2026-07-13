# Calculadora interactiva de distribuciones de probabilidad

Aplicación web educativa para calcular probabilidades, determinar cuantiles y visualizar distribuciones de probabilidad discretas y continuas.

## ¿Para qué sirve?

La aplicación permite trabajar de manera interactiva con distribuciones de probabilidad y relacionar tres elementos fundamentales:

* los parámetros que definen una distribución;
* el evento de interés;
* la probabilidad o el cuantil asociado.

Puede utilizarse para resolver ejercicios, comprobar cálculos, explorar el efecto de los parámetros y reforzar la interpretación gráfica de probabilidades y cuantiles.

## Modos de cálculo

### 1. Probabilidad a partir de un evento

En este modo se especifica un evento y la aplicación calcula su probabilidad.

Se encuentran disponibles los siguientes tipos de eventos:

* `P(X ≤ x)`: probabilidad acumulada hasta un valor determinado;
* `P(X ≥ x)`: probabilidad de obtener un valor igual o mayor que un límite;
* `P(X = x)`: probabilidad de un valor puntual;
* `P(a ≤ X ≤ b)`: probabilidad comprendida entre dos límites.

En las distribuciones continuas, la probabilidad de un valor puntual es cero. En las distribuciones discretas, el evento `P(X = x)` solo puede tener probabilidad positiva cuando `x` pertenece al soporte de la distribución.

### 2. Evento a partir de una probabilidad

En este modo se introduce una probabilidad objetivo y la aplicación determina el cuantil o límite correspondiente.

Permite resolver:

* `P(X ≤ x) = p`;
* `P(X ≥ x) = p`;
* `P(a ≤ X ≤ b) = p`, con `a` conocido;
* `P(a ≤ X ≤ b) = p`, con `b` conocido.

Este modo resulta útil para obtener valores críticos, percentiles, límites de aceptación y puntos de corte.

## Distribuciones disponibles

### Distribuciones discretas

* Bernoulli
* Binomial
* Poisson
* Binomial negativa
* Geométrica
* Hipergeométrica

### Distribuciones continuas

* Normal
* Exponencial
* Uniforme
* Gamma
* Chi-cuadrado
* t de Student
* F de Fisher

Para cada distribución se presenta una breve descripción, su función matemática y los parámetros necesarios para definirla.

## Cómo utilizar la calculadora

1. Abra el cuadro **Configuración general**.
2. Seleccione una distribución.
3. Seleccione el modo de cálculo:
   * probabilidad a partir de un evento; o
   * evento a partir de una probabilidad.
4. Introduzca los parámetros de la distribución.
5. Seleccione el tipo de evento.
6. Complete los límites o la probabilidad solicitada.
7. Presione **Calcular y graficar** o **Calcular evento y graficar**.

Los campos de entrada se actualizan automáticamente cuando se cambia el tipo de evento.

## Presentación de los resultados

La aplicación destaca primero el resultado principal solicitado.

Cuando se calcula una probabilidad, se muestra:

* la probabilidad calculada;
* su equivalencia porcentual;
* la expresión matemática del evento.

Cuando se calcula un evento a partir de una probabilidad, se muestra:

* el cuantil calculado;
* el límite inferior o superior obtenido, según corresponda;
* la probabilidad objetivo;
* la probabilidad efectivamente alcanzada.

Como **Información de la distribución**, también se presentan:

* la media;
* la varianza.

## Visualización gráfica

Cada cálculo genera dos gráficos complementarios.

### Función de densidad

Representa cómo se distribuye la probabilidad entre los posibles valores de la variable aleatoria.

La región correspondiente al evento calculado aparece resaltada, lo que permite relacionar el resultado numérico con el área o los valores considerados.

En el curso se utiliza la expresión **función de densidad** tanto para distribuciones continuas como discretas.

### Función de distribución acumulada

Representa la probabilidad acumulada `P(X ≤ x)` en función de `x`.

Este gráfico facilita la interpretación de:

* probabilidades acumuladas;
* cuantiles;
* percentiles;
* colas de una distribución;
* límites asociados a una probabilidad.

Los gráficos se muestran uno debajo del otro para aprovechar mejor el ancho disponible y facilitar su lectura.

## Consideraciones sobre distribuciones discretas

En una distribución discreta, la función acumulada avanza mediante saltos. Por este motivo, no siempre existe un valor entero que produzca exactamente la probabilidad objetivo introducida.

Cuando esto ocurre, la aplicación muestra un evento alcanzable y señala la diferencia entre la probabilidad solicitada y la obtenida.

Los límites no enteros se interpretan respetando el soporte discreto. Por ejemplo, un evento como `X ≤ 3.7` incluye los valores enteros hasta `3`.

## Consideraciones sobre los gráficos

El cálculo numérico utiliza la distribución completa. Para mantener una visualización clara, los gráficos representan un rango central adecuado de la distribución.

Si un límite se encuentra fuera de ese rango visual, la aplicación lo informa. Esto no altera la probabilidad calculada.

## Acceso a la aplicación

La calculadora está disponible en:

**https://calculadoradedistribuciones.streamlit.app/**

No requiere instalación. Puede utilizarse directamente desde un navegador web en computadora, tableta o teléfono móvil.
